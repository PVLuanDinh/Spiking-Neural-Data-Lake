//! Phase 2 of the Rust port: `PrefrontalExecutive._decide_spiking` (`faculties/prefrontal_executive.py`), the
//! Wang-2002 spiking winner-take-all decision attractor, line for line: the same draw order on the instance's
//! CPython MT19937 stream ([`PyRandom`]), the same float operation order, `_argmax` ties to the lowest index, and
//! CPython 3.12+'s compensated `sum()` for the inhibitory drive ([`builtin_sum`]). Judged call-for-call and
//! state-for-state against `tests/fixtures_pfc.json` (recorded from the deleted Python body; lane pfc-p2).
use crate::rng::mutant;
use crate::rng::pyrandom::PyRandom;

/// The instance constants `_decide_spiking` reads (derived once in `PrefrontalExecutive.__init__`; the binding
/// passes them per call so a test that pokes `pfc._theta` still sees its value).
#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub n_dec: usize,
    pub n_inh: usize,
    pub t_max: usize,
    pub md: f64,
    pub g_ext: f64,
    pub b0: f64,
    pub w_plus: f64,
    pub w_inh: f64,
    pub w_ie: f64,
    pub w_cross: f64,
    pub trace_decay: f64,
    pub theta: f64,
    pub noise_amp: f64,
    pub init_v: f64,
    /// `n_dec * n_options` fixed per-neuron biases (`_dec_bias`), indexed `i * n_dec + k`.
    pub dec_bias: Vec<f64>,
}

// P8 reds (test-only mutants on the real code path; the kit's `mutant` switch, bits above the kit's own).
pub(crate) const EXTRA_DRAW: u8 = 32; // one extra draw before the race (the Phase 0 constructed red)

/// CPython >= 3.12 `sum()` over floats from its int `0` start: Neumaier-compensated, the compensation added at
/// the end when non-zero and finite (`Python/bltinmodule.c`, `builtin_sum_impl`). Kept because it IS what the Python
/// body computed, not because the fixture needs it: a plain left fold was measured as a red and differed on 0 of 437
/// races (lane pfc-p2 R2, a NULL), so this is semantics, not a defence the fixture can see.
pub fn builtin_sum(xs: &[f64]) -> f64 {
    let (mut f, mut c) = (0.0f64, 0.0f64);
    for &x in xs {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// `_argmax`: index of the largest value, ties to the lowest index.
fn argmax(values: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in values.iter().enumerate() {
        if v > values[best] {
            best = i;
        }
    }
    best
}

/// Whether `decide_spiking` races at all: some drive is `> 0` (NaN is not). The binding checks its
/// `dec_bias` length only when this holds, so no-evidence calls of any length stay `(0, 1)` as the
/// Python body they replaced returned before it indexed `_dec_bias`.
pub fn has_evidence(evidence: &[f64]) -> bool {
    evidence.iter().any(|&e| e > 0.0)
}

/// One spiking winner-take-all decision over `evidence` (>= 0 drive per option) on `rng`, the instance's
/// `random.Random` stream. Returns `(winner_index, rt_steps)`; no evidence -> `(0, 1)` without a draw.
pub fn decide_spiking(p: &Params, evidence: &[f64], rng: &mut PyRandom) -> (usize, usize) {
    if !has_evidence(evidence) {
        return (0, 1); // no evidence -> no race (tie -> index 0)
    }
    let n = evidence.len();
    let mut e_ref = 0.0;
    for &e in evidence {
        if e > e_ref {
            e_ref = e;
        }
    }
    if mutant(EXTRA_DRAW) {
        rng.random();
    }
    let scaled: Vec<f64> = evidence
        .iter()
        .map(|&e| (if e > 0.0 { e } else { 0.0 }) / e_ref)
        .collect();
    let mut sum_scaled = 0.0;
    for &s in &scaled {
        sum_scaled += s;
    }
    let (nd, ni) = (p.n_dec, p.n_inh);
    // feed-forward cross-inhibition: each pool is suppressed by the MEAN drive of the others
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let cross: Vec<f64> = (0..n)
        .map(|i| p.w_cross * (sum_scaled - scaled[i]) / denom)
        .collect();
    // LIF membranes: seed-dependent random initial charge, option pools first (i-major), then the inhibitory pool
    let mut vd: Vec<f64> = Vec::with_capacity(n * nd);
    for _ in 0..n * nd {
        vd.push(p.init_v * rng.random());
    }
    let mut vi: Vec<f64> = (0..ni).map(|_| p.init_v * rng.random()).collect();
    let mut trace = vec![0.0f64; n]; // per option-pool firing-rate proxy
    let mut trace_i = 0.0f64; // inhibitory-pool firing-rate proxy
    let mut spk = vec![0usize; n];
    let mut winner = None;
    let mut rt = p.t_max;
    for t in 1..=p.t_max {
        let inh = p.w_inh * trace_i; // shared spiking inhibition this step
        for i in 0..n {
            let cur = p.g_ext * scaled[i] - cross[i] + p.w_plus * trace[i] - inh
                + p.b0
                + p.noise_amp * (rng.random() - 0.5);
            let base = i * nd;
            let mut s = 0;
            for k in 0..nd {
                let mut v = p.md * vd[base + k] + cur + p.dec_bias[base + k]; // LIF: leak + integrate
                if v >= 1.0 {
                    s += 1; // threshold -> discrete spike -> reset
                    v = 0.0;
                }
                vd[base + k] = v;
            }
            spk[i] = s;
        }
        // option pools excite the inhibitory pool (`sum(trace)`: CPython's compensated builtin)
        let exc = p.w_ie * builtin_sum(&trace);
        let mut si = 0;
        for v in vi.iter_mut() {
            let mut nv = p.md * *v + exc + p.noise_amp * (rng.random() - 0.5);
            if nv >= 1.0 {
                si += 1;
                nv = 0.0;
            }
            *v = nv;
        }
        for i in 0..n {
            trace[i] =
                p.trace_decay * trace[i] + (1.0 - p.trace_decay) * (spk[i] as f64 / nd as f64);
        }
        trace_i = p.trace_decay * trace_i + (1.0 - p.trace_decay) * (si as f64 / ni as f64);
        let best = argmax(&trace);
        if trace[best] >= p.theta {
            winner = Some(best); // first pool to sustained high firing wins
            rt = t;
            break;
        }
    }
    (winner.unwrap_or_else(|| argmax(&trace)), rt) // timed out -> the strongest pool so far
}

#[cfg(test)]
mod tests {
    //! Parity against the Python oracle (`tests/fixtures_pfc.json`: the 5 Phase 0 `cnt220` full cases, all the
    //! zero-evidence no-draw path, plus three chains of 460 calls: 23 zero-path, 206 timeouts) and the two
    //! pre-registered red (REPORT.md 1.2 C2, C2', R1 of lane pfc-p2; R2, the plain-sum mutant, measured 0/437 and
    //! was removed as a NULL -- see REPORT.md).
    use super::*;
    use crate::rng::crc32;
    use crate::rng::mt19937::N;
    use crate::rng::mutants;
    use crate::rng::pyrandom::State;
    use serde_json::Value;

    const FIX: &str = include_str!("../tests/fixtures_pfc.json");

    fn state(v: &Value) -> State {
        let w: Vec<u32> = v
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(w.len(), N + 1);
        let mut words = [0u32; N];
        words.copy_from_slice(&w[..N]);
        State {
            words,
            pos: w[N] as usize,
            gauss_next: None,
        }
    }

    /// `zlib.crc32(struct.pack("<625I", *getstate()[1]))`, the fixture's post-state digest.
    fn crc_state(st: &State) -> u32 {
        let mut b = Vec::with_capacity(4 * (N + 1));
        for w in st.words.iter().chain(std::iter::once(&(st.pos as u32))) {
            b.extend_from_slice(&w.to_le_bytes());
        }
        crc32(&b)
    }

    fn floats(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    fn params(v: &Value) -> Params {
        // the Phase 0 full cases carry the Python attribute names (`_n_dec`, ...); the chains the bare ones
        let g = |k: &str| v.get(k).or_else(|| v.get(format!("_{k}"))).unwrap().clone();
        let f = |k: &str| g(k).as_f64().unwrap();
        let u = |k: &str| g(k).as_u64().unwrap() as usize;
        Params {
            n_dec: u("n_dec"),
            n_inh: u("n_inh"),
            t_max: u("t_max"),
            md: f("md"),
            g_ext: f("g_ext"),
            b0: f("b0"),
            w_plus: f("w_plus"),
            w_inh: f("w_inh"),
            w_ie: f("w_ie"),
            w_cross: f("w_cross"),
            trace_decay: f("trace_decay"),
            theta: f("theta"),
            noise_amp: f("noise_amp"),
            init_v: f("init_v"),
            dec_bias: floats(&g("dec_bias")),
        }
    }

    fn rng_at(v: &Value) -> PyRandom {
        let mut rng = PyRandom::from_u128(0);
        rng.setstate(&state(v));
        rng
    }

    fn out(v: &Value) -> (usize, usize) {
        (
            v[0].as_u64().unwrap() as usize,
            v[1].as_u64().unwrap() as usize,
        )
    }

    #[test]
    fn has_evidence_is_the_no_race_predicate() {
        assert!(!has_evidence(&[]));
        assert!(!has_evidence(&[0.0; 14]));
        assert!(!has_evidence(&[-1.0, f64::NAN]));
        assert!(has_evidence(&[0.0, 1e-300]));
    }

    #[test]
    fn the_five_phase0_full_cases_reproduce_from_their_pre_state() {
        let fix: Value = serde_json::from_str(FIX).unwrap();
        let full = fix["full"].as_array().unwrap();
        assert_eq!(full.len(), 5);
        for f in full {
            let mut rng = rng_at(&f["pre"][1]);
            assert_eq!(
                decide_spiking(&params(&f["params"]), &floats(&f["evidence"]), &mut rng),
                out(&f["out"])
            );
            assert_eq!(rng.getstate(), state(&f["post"][1]));
        }
    }

    #[test]
    fn the_oracle_chains_match_call_for_call_and_state_for_state() {
        let fix: Value = serde_json::from_str(FIX).unwrap();
        let (mut calls, mut zero, mut timeouts) = (0u64, 0u64, 0u64);
        for ch in fix["chains"].as_array().unwrap() {
            let p = params(&ch["params"]);
            let mut rng = rng_at(&ch["state0"]);
            for (i, c) in ch["calls"].as_array().unwrap().iter().enumerate() {
                let e = floats(&c["e"]);
                let got = decide_spiking(&p, &e, &mut rng);
                assert_eq!(
                    got,
                    out(&c["out"]),
                    "chain {} call {i}: (winner, rt)",
                    ch["name"]
                );
                assert_eq!(
                    u64::from(crc_state(&rng.getstate())),
                    c["crc"].as_u64().unwrap(),
                    "chain {} call {i}: post-state",
                    ch["name"]
                );
                calls += 1;
                zero += u64::from(e.iter().cloned().fold(0.0, f64::max) <= 0.0);
                timeouts += u64::from(got.1 == p.t_max);
            }
            assert_eq!(
                rng.getstate(),
                state(&ch["state_end"]),
                "chain {} final state",
                ch["name"]
            );
        }
        // anti-vacuity: the replay walked the counts the generator wrote
        let counts = &fix["counts"];
        assert_eq!(calls, counts["calls"].as_u64().unwrap());
        assert_eq!(zero, counts["zero_path"].as_u64().unwrap());
        assert_eq!(timeouts, counts["timeouts"].as_u64().unwrap());
        assert_eq!((calls, zero, timeouts), (460, 23, 206));
    }

    /// Runs every chain call from the ORACLE pre-state with `bit` on; returns (calls whose post-state differs,
    /// zero-path calls whose post-state differs, calls whose (winner, rt) differs, non-zero-path calls).
    fn red(bit: u8) -> (u64, u64, u64, u64) {
        let fix: Value = serde_json::from_str(FIX).unwrap();
        let (mut diff, mut zero_diff, mut out_diff, mut nonzero) = (0u64, 0u64, 0u64, 0u64);
        for ch in fix["chains"].as_array().unwrap() {
            let p = params(&ch["params"]);
            let mut rng = rng_at(&ch["state0"]);
            for c in ch["calls"].as_array().unwrap() {
                let e = floats(&c["e"]);
                let is_zero = e.iter().cloned().fold(0.0, f64::max) <= 0.0;
                nonzero += u64::from(!is_zero);
                let pre = rng.getstate();
                mutants::set(bit);
                let got = decide_spiking(&p, &e, &mut rng);
                mutants::set(0);
                let d = u64::from(crc_state(&rng.getstate())) != c["crc"].as_u64().unwrap();
                diff += u64::from(d);
                zero_diff += u64::from(d && is_zero);
                out_diff += u64::from(got != out(&c["out"]));
                rng.setstate(&pre);
                assert_eq!(decide_spiking(&p, &e, &mut rng), out(&c["out"])); // back on the oracle trajectory
            }
        }
        (diff, zero_diff, out_diff, nonzero)
    }

    #[test]
    fn red_one_extra_draw_changes_the_post_state_of_every_race() {
        let (diff, zero_diff, out_diff, nonzero) = red(EXTRA_DRAW);
        println!(
            "EXTRA_DRAW: post-state differs {diff}/{nonzero} races, (winner, rt) differs {out_diff}"
        );
        assert_eq!((diff, zero_diff, nonzero), (437, 0, 437));
    }
}
