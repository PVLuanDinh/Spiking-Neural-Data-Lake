//! The ONE fixture comparator (RNG-KIT.md section 2, criteria C1-C8): every claimed kit section is run
//! from a fresh engine per seed and compared, outputs AND post-state, with `tests/fixtures_rng.json`
//! (real CPython 3.14.6t `random` + numpy 2.5.1). A cell = one `(family, seed, section)`; it fails when
//! either half differs. Encoding is the kit's: floats as Python `float.hex()`, ints past 2^53 as decimal
//! strings, post-state digests = sha256 of the kit's canonical JSON. The OPEN sections ([`UNCLAIMED`])
//! are listed, never compared, never counted as passing. The `#[cfg(test)]` mutants (P8 reds) run
//! through this same function.
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::lcg::Lcg;
use super::legacy::RandomState;
use super::pcg64::{Generator, Pcg64State};
use super::pyrandom::PyRandom;
use super::seedseq::SeedSequence;
use super::{crc32, words_from_decimal};

/// Fixture sections this port does not claim: OPEN-libm (`gauss`, `expovariate`, `randn`) and
/// OPEN-ziggurat (`normal`, `standard_normal`), RNG-KIT.md section 3.
pub const UNCLAIMED: &[&str] = &[
    "gauss_0_1",
    "gauss_2_05",
    "expovariate1",
    "normal_0_1",
    "standard_normal",
    "randn_31",
];
/// sha256 of the kit's `fixtures_rng.json` (RNG-KIT.md), as the kit wrote it.
pub const KIT_SHA: &str = "1515bc47f7e3c598f5faa52f96ba212f6e635170a07b56f20dcf42ef0f381997";
/// `carried_state_campaign.SHUFFLE_TAG` ("SHUF"), the list-entropy middle word.
pub const SHUF_TAG: u128 = 0x5348_5546;

#[derive(Debug, Default)]
pub struct Report {
    /// Cells compared.
    pub claimed: usize,
    /// Cells whose outputs or post-state differ from the fixture.
    pub failed: Vec<String>,
    /// OPEN cells listed in the fixture and skipped here.
    pub unclaimed: Vec<String>,
}

enum Out {
    F(f64),
    I(i64),
    U(u128),
}

/// Python `float.hex()`.
pub fn py_hex(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let bits = f.to_bits();
    let sign = if bits >> 63 == 1 { "-" } else { "" };
    let exp = ((bits >> 52) & 0x7ff) as i32;
    let mant = bits & ((1u64 << 52) - 1);
    match (exp, mant) {
        (0, 0) => format!("{sign}0x0.0p+0"),
        (0, m) => format!("{sign}0x0.{m:013x}p-1022"),
        (e, m) => format!("{sign}0x1.{m:013x}p{:+}", e - 1023),
    }
}

/// True when `bytes` is the kit's `fixtures_rng.json` (sha256 `KIT_SHA`), under either checkout line
/// ending: the kit wrote the file in Windows text mode (CRLF) and the repo stores it LF-normalised, so
/// a Linux checkout differs from the kit's bytes only by the newline (a `-text` attribute, requested
/// from the integrator, makes the bytes identical everywhere; until then both forms are accepted).
pub fn is_kit_fixture(bytes: &[u8]) -> bool {
    if sha256_hex(bytes) == KIT_SHA {
        return true;
    }
    let mut crlf = Vec::with_capacity(bytes.len() + bytes.len() / 32);
    let mut prev = 0u8;
    for &b in bytes {
        if b == b'\n' && prev != b'\r' {
            crlf.push(b'\r');
        }
        crlf.push(b);
        prev = b;
    }
    sha256_hex(&crlf) == KIT_SHA
}

pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// The kit's `E()`: ints with |x| > 2^53 become decimal strings.
fn enc_u(x: u128) -> Value {
    if x > 1u128 << 53 {
        Value::String(x.to_string())
    } else {
        Value::from(x as u64)
    }
}

fn enc(outs: &[Out]) -> Value {
    Value::Array(
        outs.iter()
            .map(|o| match *o {
                Out::F(f) => Value::String(py_hex(f)),
                Out::I(i) if i.unsigned_abs() > 1u64 << 53 => Value::String(i.to_string()),
                Out::I(i) => Value::from(i),
                Out::U(u) => enc_u(u),
            })
            .collect(),
    )
}

/// `kit.dig`: sha256 of `json.dumps(obj, separators=(",", ":"))`.
fn dig(v: &Value) -> String {
    sha256_hex(v.to_string().as_bytes())
}

/// `kit.std_post`: `sha256([3, [words..., pos], gauss_hex|null])`, `pos`, `gauss_next`.
fn std_post(r: &PyRandom) -> Value {
    let st = r.getstate();
    let mut internal: Vec<Value> = st.words.iter().map(|&w| Value::from(w)).collect();
    internal.push(Value::from(st.pos));
    let g = st.gauss_next.map(py_hex);
    json!({"sha256": dig(&json!([3, internal, g])), "pos": st.pos, "gauss_next": g})
}

/// `kit.np_post`: the full PCG64 state.
fn np_post(st: Pcg64State) -> Value {
    json!({
        "state": enc_u(st.state),
        "inc": enc_u(st.inc),
        "has_uint32": u8::from(st.has_uint32),
        "uinteger": enc_u(u128::from(st.uinteger)),
    })
}

/// `kit.leg_post`: `sha256([keys, pos, has_gauss, cached_hex])`, `pos`, `has_gauss`, `cached`.
fn leg_post(r: &RandomState) -> Value {
    let (keys, pos, hg, cached) = r.state();
    let keys: Vec<Value> = keys.iter().map(|&w| Value::from(w)).collect();
    let hg = u8::from(hg);
    let cached = py_hex(cached);
    json!({"sha256": dig(&json!([keys, pos, hg, cached])), "pos": pos, "has_gauss": hg, "cached": cached})
}

fn flat<I: IntoIterator<Item = usize>>(rounds: usize, mut one: impl FnMut() -> I) -> Vec<Out> {
    (0..rounds)
        .flat_map(|_| {
            one()
                .into_iter()
                .map(|i| Out::I(i as i64))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn std_section(name: &str, r: &mut PyRandom) -> Option<Vec<Out>> {
    const KS: [u32; 6] = [1, 7, 32, 33, 64, 128];
    Some(match name {
        "random" => (0..64).map(|_| Out::F(r.random())).collect(),
        "getrandbits32" => (0..64).map(|_| Out::U(r.getrandbits(32))).collect(),
        "getrandbits_mixed" => (0..32).map(|i| Out::U(r.getrandbits(KS[i % 6]))).collect(),
        "randrange10" => (0..32).map(|_| Out::I(r.randrange(0, 10, 1))).collect(),
        "randrange_step" => (0..32).map(|_| Out::I(r.randrange(5, 1000, 7))).collect(),
        "randrange_big" => (0..32)
            .map(|_| Out::I(r.randrange(0, 3 << 38, 1)))
            .collect(),
        "randint_0_1" => (0..32).map(|_| Out::I(r.randint(0, 1))).collect(),
        "randint_m5_5" => (0..32).map(|_| Out::I(r.randint(-5, 5))).collect(),
        "uniform_m1_3" => (0..32).map(|_| Out::F(r.uniform(-1.0, 3.0))).collect(),
        "choice37" => (0..32).map(|_| Out::I(r.choice_index(37) as i64)).collect(),
        "choices_k4" => flat(8, || r.choices_indices(37, 4)),
        "shuffle16" => flat(8, || {
            let mut a: Vec<usize> = (0..16).collect();
            r.shuffle(&mut a);
            a
        }),
        "sample_pool_n20k5" => flat(32, || r.sample_indices(20, 5)),
        "sample_pool_n40k30" => flat(4, || r.sample_indices(40, 30)),
        "sample_set_n1000k5" => flat(32, || r.sample_indices(1000, 5)),
        "getstate_setstate" => {
            for _ in 0..5 {
                r.random();
            }
            let st = r.getstate();
            let mut out: Vec<Out> = (0..16).map(|_| Out::F(r.random())).collect();
            r.setstate(&st);
            out.extend((0..16).map(|_| Out::F(r.random())));
            out
        }
        _ => return None,
    })
}

fn np_section(name: &str, g: &mut Generator) -> Option<Vec<Out>> {
    Some(match name {
        "random" => (0..64).map(|_| Out::F(g.random())).collect(),
        "integers_0_2p63" => (0..64)
            .map(|_| Out::U(g.integers_u64(0, 1 << 63).into()))
            .collect(),
        "integers_0_96" => (0..32).map(|_| Out::I(g.integers(0, 96))).collect(),
        "integers_0_3x2p62" => (0..32)
            .map(|_| Out::U(g.integers_u64(0, 3 << 62).into()))
            .collect(),
        "integers_0_1_nodraw" => {
            let mut out: Vec<Out> = (0..32).map(|_| Out::I(g.integers(0, 1))).collect();
            out.push(Out::F(g.random()));
            out
        }
        "uniform_m1_3" => (0..32).map(|_| Out::F(g.uniform(-1.0, 3.0))).collect(),
        "permutation20" => flat(4, || g.permutation(20)),
        "shuffle16" => flat(4, || {
            let mut a: Vec<usize> = (0..16).collect();
            g.shuffle(&mut a);
            a
        }),
        "choice37" => (0..32).map(|_| Out::U(g.choice(37).into())).collect(),
        "interleave_u32_u64" => (0..8)
            .flat_map(|_| {
                [
                    Out::I(g.integers(0, 96)),
                    Out::I(g.integers(0, 96)),
                    Out::I(g.integers(0, 96)),
                    Out::F(g.random()),
                ]
            })
            .collect(),
        _ => return None,
    })
}

fn leg_section(name: &str, r: &mut RandomState) -> Option<Vec<Out>> {
    Some(match name {
        "rand" => (0..64).map(|_| Out::F(r.rand())).collect(),
        "permutation20" => flat(4, || r.permutation(20)),
        "choice37" => (0..32).map(|_| Out::I(i64::from(r.choice(37)))).collect(),
        _ => return None,
    })
}

fn u64_of(words: &[u32]) -> u64 {
    u64::from(words[0]) | (u64::from(*words.get(1).unwrap_or(&0)) << 32)
}

impl Report {
    fn cell(&mut self, name: String, got: Value, want: &Value) {
        self.claimed += 1;
        if got != *want {
            self.failed.push(name);
        }
    }

    /// Run `sections` (fixture object) through `run_one` with a fresh engine per section; `post` is
    /// read after the section's draws.
    fn sections<E>(
        &mut self,
        prefix: &str,
        sections: &Map<String, Value>,
        mut fresh: impl FnMut() -> E,
        run_one: impl Fn(&str, &mut E) -> Option<Vec<Out>>,
        post: impl Fn(&E) -> Value,
    ) {
        for (name, want) in sections {
            let cell = format!("{prefix}/{name}");
            if UNCLAIMED.contains(&name.as_str()) {
                self.unclaimed.push(cell);
                continue;
            }
            let mut e = fresh();
            let got = match run_one(name, &mut e) {
                Some(outs) => json!({"out": enc(&outs), "post": post(&e)}),
                None => Value::Null,
            };
            self.cell(cell, got, want);
        }
    }
}

fn obj<'a>(v: &'a Value, key: &str) -> &'a Map<String, Value> {
    v.get(key)
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("fixture: `{key}` missing or not an object"))
}

/// Compare every claimed cell of `fixture_json` (the kit's `fixtures_rng.json` text).
pub fn run(fixture_json: &str) -> Report {
    let fx: Value = serde_json::from_str(fixture_json).expect("fixture is valid JSON");
    let mut rep = Report::default();

    for (seed, fs) in obj(&fx, "stdlib") {
        let words = words_from_decimal(seed).expect("stdlib seed is decimal");
        let r0 = PyRandom::from_words(&words);
        let st = r0.getstate();
        let got = json!({"version": 3, "mt": st.words.to_vec(), "pos": st.pos, "gauss_next": null});
        rep.cell(
            format!("stdlib/{seed}/state_after_seed"),
            got,
            &fs["state_after_seed"],
        );
        rep.sections(
            &format!("stdlib/{seed}"),
            obj(fs, "sections"),
            || PyRandom::from_words(&words),
            std_section,
            std_post,
        );
    }

    for (tag, fs) in obj(&fx, "stdlib_str_seed") {
        let mut r = PyRandom::from_str_seed(tag);
        let sha = std_post(&r)["sha256"].clone();
        let random16 = enc(&(0..16).map(|_| Out::F(r.random())).collect::<Vec<_>>());
        let rr: Vec<Value> = (0..16)
            .map(|_| Value::from(r.randrange(0, 10, 1)))
            .collect();
        let got = json!({"state_after_seed_sha256": sha, "random16": random16,
                         "randrange10_16": rr, "post": std_post(&r)});
        rep.cell(format!("stdlib_str_seed/{tag}"), got, fs);
    }

    for (tag, fs) in obj(&fx, "cortex_rng") {
        let c = crc32(tag.as_bytes());
        let mut r = PyRandom::from_u128(u128::from(c));
        let ri: Vec<Value> = (0..32).map(|_| Value::from(r.randint(0, 1))).collect();
        let got = json!({"crc32": c, "randint_0_1": ri, "post": std_post(&r)});
        rep.cell(format!("cortex_rng/{tag}"), got, fs);
    }

    for (seed, fs) in obj(&fx, "numpy_generator") {
        let words = words_from_decimal(seed).expect("numpy seed is decimal");
        let p = format!("numpy_generator/{seed}");
        let ss = SeedSequence::new(words.clone());
        let g4 = |ss: &SeedSequence| {
            Value::Array(
                ss.generate_u64(4)
                    .into_iter()
                    .map(|x| enc_u(x.into()))
                    .collect(),
            )
        };
        rep.cell(
            format!("{p}/seedseq_state4_u64"),
            g4(&ss),
            &fs["seedseq_state4_u64"],
        );
        let st = Generator::from_words(words.clone()).state();
        rep.cell(
            format!("{p}/state_after_seed"),
            np_post(st),
            &fs["state_after_seed"],
        );
        let seed_words = words.clone();
        let le: Map<String, Value> = (0..2u32)
            .map(|k| {
                let mut e = seed_words.clone();
                e.extend(SeedSequence::from_list(&[SHUF_TAG, k.into()]).entropy());
                (k.to_string(), g4(&SeedSequence::new(e)))
            })
            .collect();
        rep.cell(
            format!("{p}/seedseq_list_entropy_state4_u64"),
            Value::Object(le),
            &fs["seedseq_list_entropy_state4_u64"],
        );
        let kids: Vec<Value> = SeedSequence::new(words.clone())
            .spawn(3)
            .iter()
            .map(g4)
            .collect();
        rep.cell(
            format!("{p}/spawn3_state4_u64"),
            Value::Array(kids),
            &fs["spawn3_state4_u64"],
        );
        let sections = obj(fs, "sections");
        let hinge = PyRandom::from_words(&words).getrandbits(128);
        rep.sections(
            &p,
            sections,
            || Generator::from_words(words.clone()),
            |name, g| match name {
                "from_getrandbits128" => {
                    *g = Generator::from_u128(hinge);
                    Some((0..16).map(|_| Out::F(g.random())).collect())
                }
                _ => np_section(name, g),
            },
            |g| np_post(g.state()),
        );
    }

    for (seed, fs) in obj(&fx, "numpy_legacy") {
        let s: u32 = seed.parse().expect("legacy seed < 2^32");
        rep.cell(
            format!("numpy_legacy/{seed}/state_after_seed"),
            leg_post(&RandomState::new(s)),
            &fs["state_after_seed"],
        );
        rep.sections(
            &format!("numpy_legacy/{seed}"),
            obj(fs, "sections"),
            || RandomState::new(s),
            leg_section,
            leg_post,
        );
    }

    for (seed, fs) in obj(&fx, "mathutils_rng") {
        let words = words_from_decimal(seed).expect("mathutils seed is decimal");
        let low = u64_of(&words);
        let mut m = Lcg::new(if seed.starts_with('-') {
            low.wrapping_neg()
        } else {
            low
        });
        let s0 = m.state().to_string();
        let r64 = enc(&(0..64).map(|_| Out::F(m.random())).collect::<Vec<_>>());
        let got =
            json!({"state_after_seed": s0, "random64": r64, "state_after": m.state().to_string()});
        rep.cell(format!("mathutils_rng/{seed}"), got, fs);
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::super::{BAD_INC, NO_U32_BUFFER, RNG0_DRAWS, SINGLE_WORD_SEED, SWAP53, mutants};
    use super::{KIT_SHA, Report, is_kit_fixture, py_hex, run, sha256_hex};

    const FIX: &str = include_str!("../../tests/fixtures_rng.json");

    fn with(bits: u8) -> Report {
        mutants::set(bits);
        let r = run(FIX);
        mutants::set(0);
        r
    }

    fn red(name: &str, bits: u8, expect: usize) {
        let r = with(bits);
        let mut secs: Vec<&str> = r
            .failed
            .iter()
            .map(|c| c.rsplit('/').next().unwrap())
            .collect();
        secs.sort_unstable();
        secs.dedup();
        println!(
            "RED {name}: {} failing cells (pre-registered {expect}) sections={secs:?}",
            r.failed.len()
        );
        assert_eq!(r.failed.len(), expect, "{name}: {:?}", r.failed);
    }

    #[test]
    fn float_hex_is_pythons() {
        assert_eq!(py_hex(0.0), "0x0.0p+0");
        assert_eq!(py_hex(1.0), "0x1.0000000000000p+0");
        assert_eq!(py_hex(-0.5), "-0x1.0000000000000p-1");
        assert_eq!(py_hex(5e-324), "0x0.0000000000001p-1022");
    }

    #[test]
    fn fixture_copy_is_the_kits() {
        assert!(
            is_kit_fixture(FIX.as_bytes()),
            "sha256 {} != kit {KIT_SHA}",
            sha256_hex(FIX.as_bytes())
        );
        assert!(
            !is_kit_fixture(&FIX.as_bytes()[1..]),
            "a changed copy must be refused"
        );
        // both checkout forms of the same bytes are the kit's
        let lf = FIX.replace("\r\n", "\n");
        let crlf = lf.replace('\n', "\r\n");
        assert!(is_kit_fixture(lf.as_bytes()) && is_kit_fixture(crlf.as_bytes()));
    }

    #[test]
    fn green_every_claimed_cell() {
        let r = with(0);
        println!(
            "GREEN: claimed {} failed {} unclaimed {}",
            r.claimed,
            r.failed.len(),
            r.unclaimed.len()
        );
        assert_eq!(r.failed, Vec::<String>::new());
        assert_eq!(r.claimed, 254);
        assert_eq!(r.unclaimed.len(), 39);
    }

    #[test]
    fn red_swap53() {
        red("swap53", SWAP53, 35);
    }

    #[test]
    fn red_single_word_seed() {
        red("single_word_seed", SINGLE_WORD_SEED, 57);
    }

    #[test]
    fn red_bad_inc() {
        red("bad_inc", BAD_INC, 84);
    }

    #[test]
    fn red_rng0_draws() {
        red("rng0_draws", RNG0_DRAWS, 7);
    }

    #[test]
    fn red_no_u32_buffer() {
        red("no_u32_buffer", NO_U32_BUFFER, 35);
    }
}
