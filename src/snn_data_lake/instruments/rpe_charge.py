"""
rpe_charge -- the E1b harness: does the RPE harm replicate? The production tier decides (addendum 3), no-encoder beside.

Binds to specs/022-unified-brain-arc/evidence/e1b-rpe-harm.md (FROZEN at its commit, Revision 2). That text owns every
bar, constant and rule; this module only executes them. It ports E1's lane scripts (e1_probe / e1_replay / e1_compare) into
ONE governed module and adds Revision 2's changes: the tier pin, encoder-use counters, VOID status, the planted-HARM
stream, brain-free compare fixtures, the cross-tier rule (addendum 3: production decides) and the BRIDGE read-out.

  python -m snn_data_lake.instruments.rpe_charge                       # brain-free self-check (planted series, no brain)
  python -m snn_data_lake.instruments.rpe_charge run --arm real --seed 8 --ticks 600 --tier production \
        --home <throwaway NORD_HOME> --out <json> [--series <real json>]   # ONE arm, ONE process; the campaign runs them
  python -m snn_data_lake.instruments.rpe_charge replay <results dir> <tier> [N]
  python -m snn_data_lake.instruments.rpe_charge compare <results dir> <tier> [out.md]
  python -m snn_data_lake.instruments.rpe_charge cross <production dir> <no-encoder dir> [out.md]
  (replay / compare / cross take `--look K`, K = 1..len(CAMPAIGN_LOOKS): seeds 8 .. 8 + CAMPAIGN_LOOKS[K-1] - 1;
   addenda 2, 4, 5, 7 and 9, default look 1; every look adds `real`, and `pfrozen` up to H-P1's stop of record (its
   cap at most); `replay` writes a row for every seed of the look, and rows past H-V1's stop are written and
   binding-checked, not read)
  python -m snn_data_lake.instruments.rpe_charge fixture <logs dir>    # appends the three fixture lines to fixture.log

ARMS (E1's exact patch code; only the arm's own `_valence` instance and the bound AutonomousArm are acted on
(except `pc`, see departure (2)); foreign ValenceLearner calls are counted and passed through):
  real     nothing patched (records the series)
  vfrozen  ValenceLearner.learn -> returns self.dopamine(window, reward), NO weight update
  vshuf    ValenceLearner.learn with reward on call k := perm(r_real)[k]
  pfrozen  AutonomousArm._plan_lr := 0.0 after construction
  pshuf    AutonomousArm._pfc_bias: self._rpe swapped to perm(rpe_at_plan_real)[k] around the call
  pc       POSITIVE CONTROL: PrefrontalExecutive._decide_spiking -> constant (0, 1)

WHAT THIS ADDS (P2 graph-exists dispositions, every hit disposed):
  "neuter A/B charge probe"            tools/charge_probe.py is the drive-source neuter template (one brain, one named
      source, behavioural observables); instruments/port_charge_probe.py is the typed-port replay-twice. Neither hosts these
      arms: the RPE arms patch two LEARNING RULES (a learner's update and a scalar three-factor write) and score a held-out
      reward / prediction error across 8 seeds, not a decision flip. distinct-from both; the template's idea (neuter, read
      counted, positive control, never a partial neuter) is reused, the code is not extendable to it.
  "reward prediction error charge"     plasticity/valence_stdp.ValenceLearner and engine/autonomous_loop.AutonomousArm are the
      SUBJECTS under test, reused unmodified. faculties/dopamine_nuclei, habenula, learned_evc: other jobs, not touched.
  "valence learner neuter"             g5_part_d_reading / drive_reduction use ValenceLearner as a component; none neuters it.
  "replay learner on recorded series"  faculties/evidence_accumulation + recursive_tom replay frames through a port consumer;
      replay_hydration replays learner events. This replay re-learns a FRESH ValenceLearner on one recorded (pattern, reward)
      series and scores prediction before learning -- a different job, one short loop.
Six deliberate departures from the frozen text, disclosed here as its header asks: (1) a refusal raises `Refused`
(a SystemExit, code 3) instead of calling os._exit(3), so a test can see it; the process still exits 3 with no result file.
(2) the `pc` arm stubs PrefrontalExecutive._decide_spiking on the CLASS (E1's exact code: every executive in the brain
returns winner 0), unlike the other arms, which act on the arm's own instances only.
(3) with SINGLE_TIER_HARM_DELETES = True (addendum 3) a VOID production tier is REFUSED; the no-encoder tier is
reported beside in every state (VOID, short, pending, another tree) and never holds or vetoes the production decision
(addendum 6, owner #4616 "production tier decides"; addendum 4, re-panel B-N4; the NULL-RECORD precondition of
addendum 3 still reads the objective endpoint on both tiers, the conservative side).
(4) the compare fixtures are built in memory on every call (`fixture_runs`/`fixture_e2`), not written as `fixtures/` JSON files.
(5) a missing or wrong `tier_red.log` / `fixture.log` makes the tier VOID (harness validity) and compare still prints its report
with every status VOID, instead of refusing to print.
(6) campaign homes and output dirs must sit under an existing directory that does not contain the live lake (the shared
guard over-refuses an ancestor of ~/.snn_data_lake, the safe side).
ADDENDUM 2 (sequential design, P1, owner 2026-10-23): H-P1 and H-V1 are decided at seed-block LOOKS (8, 16, 24, 32 seeds,
cap 32) by `seq_decision` against `seq_boundaries` (Lan-DeMets O'Brien-Fleming spending, one-sided .05, non-binding futility at
conditional power < .10); a decision at a non-look seed count is refused. Looks after the first add only `real` + `pfrozen`;
the secondary endpoints stay at look 1. `replay` / `compare` / `cross` take `--look K`.
ADDENDUM 3 (owner ruling board #4906 to #4616, 2026-10-23, verbatim: "#4616: production tier decides"): the production tier
alone decides each primary's deletion (SINGLE_TIER_HARM_DELETES = True); the no-encoder tier runs on the same looks and is
reported beside it, its sign agreement and the tier effect no longer gate; a no-encoder-only crossing is a tier finding.
A futility or cap stop writes the mde at the STOPPED seed count; powered -> NULL-RECORD path, else INCONCLUSIVE (P8 item 4).
ADDENDUM 4 (owner 2026-10-24, verbatim: "Raise its cap to its power target"; re-panel round 2): every primary's design
sits in ONE place, `SEQ` (claimed effect, looks, pre-registered SD, t boundaries). H-V1's cap V1_CAP is its fixed-sample
power target under E1's measured per-seed shape times the LD-OBF inflation factor, in 8-seed looks, with shape-calibrated
t boundaries V1_SHAPE_T (the plain t boundaries crossed at 0.2185 under that shape, R-S1b); looks past 32 add `real` only.
A futility stop is taken only at a look whose mde from the pre-registered SD is at or below the claimed effect
(`futility_open`). Each E-* endpoint and the dispositions read the look-1 block at every look; H-P1 is carried from its
cap. Replay rows are bound to their real run by tree sha + series digest (`replay_binding`), and no sequential decision
is taken while a validity check is pending.
ADDENDUM 5 (P1 item 2, the owner's H-V1 ruling applied to H-P1 by the integrator): H-P1's cap P1_CAP is the first
8-seed look at which its design's power under E1's measured per-seed shape is >= .8. It keeps the normal-model boundaries
(the shape-calibrated ones are lower at every look). The campaign runs the longer of the two primaries' looks
(CAMPAIGN_LOOKS); each primary past its own cap is carried from that cap. Every look up to P1_CAP adds `real` +
`pfrozen` (this supersedes addendum 4's "Looks 5-19 add `real` only"); H-P1's looks are P1_LOOKS, not LOOKS.
ADDENDUM 6 (arc-2 panel fold, before any brain run): each primary is carried from its OWN STOP of record (`_n_owed`),
not from its cap: runs past a stop are not owed and a later look reprints the stop. The seed-sensitivity flag reads the
stopped look's `bound_t` / `k_min` and screens degeneracy on every seed of the look, on the arms that ran for it
(erratum 2026-10-25, R3-M1: was "the real/pfrozen pair", which after addendum 7's cap 40 left H-V1's seeds 48+
unscreened). A zero observed SD is never powered (`ZERO_SD`). The no-encoder tier never holds the production decision (departure 3).
ADDENDUM 7 (owner 2026-10-24, verbatim: "Smoothed shape"; before any brain run): H-P1's cap P1_CAP is re-derived under
a smoothed planted shape (variance-corrected Gaussian kernel bootstrap of E1's centred d_s, Silverman bandwidth), which
replaces addendum 5's 184. H-V1's 19 looks are now the longer, so the campaign runs CAMPAIGN_LOOKS = V1_LOOKS; looks up
to P1_CAP add `real` + `pfrozen`, later looks `real` only, and H-P1 is carried from its own stop past its cap.
ADDENDUM 9 (owner 2026-10-25, verbatim: "raise it in a new pre-data addendum"; before any brain run): P1_CAP is the
first 8-seed look at which H-P1's power is >= .8 under each of the shrunk, unshrunk and half-bandwidth smoothed shapes,
in numpy and harness. It lies past H-V1's cap again, so the campaign runs CAMPAIGN_LOOKS = P1_LOOKS (as addendum 5),
every look adds `real` and, up to H-P1's own stop (its cap at most), `pfrozen`; H-V1 is carried from its own stop.
ONE SEQUENTIAL MECHANISM (P2, re-panel B-N7): `seq_boundaries` / `seq_decision` / `seq_record` are the project's one
alpha-spending implementation (graph-exists "alpha spending": no other site). G1(c)-2 and G2, and any later
pre-registration under the P1 amendment, import them from here; moving them to a shared module is that consumer's change,
never a copy.
P5b: Python (criterion 1 fails: measurement code, no neuter of it moves a tick by >= 5%; no ported kernel exists).
Locality: the `_plan_w` update and ValenceLearner.learn are LOCAL scalar three-factor rules; this module adds no rule.
No open-weight model enters the brain: the two encoders are the project's own e-prop checkpoints already in production.
"""
# distinct-from: tools/charge_probe.py -- drive-source neuter template; no learning-rule arms, no held-out reward stats
# distinct-from: instruments/port_charge_probe.py -- typed-port replay-twice, not a learner/three-factor neuter
# distinct-from: instruments/paired_contrast.py -- its paired_t returns 0.0 on zero variance; E1's frozen rule (and the
#   reproduction of E1's REPORT numbers) needs +inf for a positive mean, so the statistic is kept here, not shared
# distinct-from: infra/_mathutils.py::pearson -- no NaN guard for a constant series

import contextlib
import copy
import functools
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time

BAR = 1.895  # one-sided .05, df 7 (S = 8); owner slot: 1.753 at S = 16
SEEDS = range(8, 16)  # E1b fresh seeds (owner); owner slot: range(8, 24)
N_SEEDS = len(SEEDS)
PSEEDS = range(8)  # the replay's planted-control seeds (tier-independent)
# "at least 5 of 8 seeds in the direction" (a tie is NOT in the direction); owner slot: 10
K_MIN = 5
EP1_MIN = 6  # E-P1: pfrozen flips above the repeat floor on >= 6 seeds; owner slot: 12
AGREE_MIN = 6  # cross-tier sign agreement of d_s; owner slot: 12
TIER_EFFECT_MIN = 6  # same-seed `real` series differ between tiers on >= 6 seeds; owner slot: 12
MDE_F = 1.00  # pre-registered mde factor (0.987 rounded up); owner slot: 0.66
# Addendum 2 (P1, owner 2026-10-23 "switch E1b to a sequential design"): H-P1 and H-V1 are decided at seed-block looks.
# addendum 2's 4-look design (seeds 8..8+n-1): the seq_* helpers' default looks; each primary's own looks are in SEQ
LOOKS = (8, 16, 24, 32)
SEQ_ALPHA = 0.05  # one-sided, per primary endpoint (Revision 2's multiplicity rule: no correction)
FUTILITY_CP = (
    0.10  # non-binding: stop for futility when conditional power under the claimed effect < .10
)
# FROZEN literals, E1 no-encoder-tier numbers: harm (reward), E-P2/H-P2 (reward), valence MAE
DH, DR, DM = 0.035, 0.0131, 0.048
# Addendum 4 (owner 2026-10-24, verbatim: "Raise its cap to its power target"): each primary's design, in ONE place.
# The pre-registered SDs are E1's NO-ENCODER paired SDs (no production-tier SD exists: the production power is borrowed).
SD_PRE = {"H-P1": 0.0695, "H-V1": 0.2040}
V1_CAP = 152  # H-V1: shape target 134 x LD-OBF inflation 1.0624 (18 looks) = 142.4 -> 144; power there .7935 < .8,
#   so one more look block: 152 (power .8155). e1b-fix-row/derive_v1.py `target` / `if` / `calib` / `power`
V1_LOOKS = tuple(range(N_SEEDS, V1_CAP + 1, N_SEEDS))
# H-V1 t boundary per look: the shape-calibrated one (R-S1b), never below the normal-model one
V1_SHAPE_T = (
    152.4982,
    42.7365,
    39.7059,
    27.9062,
    19.4013,
    15.4001,
    6.0660,
    5.5080,
    4.4073,
    3.7690,
    3.4068,
    3.1276,
    2.8925,
    2.7317,
    2.5690,
    2.4429,
    2.3370,
    2.2539,
    2.1730,
)  # derive_v1.py calib 152 (200 000 campaigns)
V1_FUTILITY_LOOKS = ()  # looks before the cap whose mde from SD_PRE["H-V1"] is <= DM: none (n=144 gives 0.0527)
# Addendum 9 (owner 2026-10-25 "raise it in a new pre-data addendum"): H-P1's power at 176 is S .9513 / U .8172 /
#   H05 .8026 (numpy; harness .9518 / .8163 / .8028); at 168 H05 gives .7998. Addendum 7's cap 40 held only for S
#   (U .6797, H05 .6871). Type I <= .0078 under every variant and the raw shape. e1b-a9-row/derive_a9.py
P1_CAP = 176
P1_LOOKS = tuple(range(N_SEEDS, P1_CAP + 1, N_SEEDS))
# looks before the cap whose mde from SD_PRE["H-P1"] is <= DH: from n=72 (0.0316; n=64 gives 0.0354),
#   e1b-smooth-row/derive_smooth.py `futility H-P1 176`
P1_FUTILITY_LOOKS = tuple(range(72, P1_CAP, N_SEEDS))
# both are 8-seed blocks from seed 8: the longer holds the other
CAMPAIGN_LOOKS = max(P1_LOOKS, V1_LOOKS, key=len)
SEQ = {
    "H-P1": {"claimed": DH, "looks": P1_LOOKS, "sd_pre": SD_PRE["H-P1"], "bounds": None},
    "H-V1": {"claimed": DM, "looks": V1_LOOKS, "sd_pre": SD_PRE["H-V1"], "bounds": V1_SHAPE_T},
}
# E1c (T122, evidence/e1c-plan-w-powered.md, frozen 082b66f8): the powered production H-P1 follow-up, ONE parameter set
#   on the same sequential path (P2: no copy). Cap 704 at 11 looks from the zero-preserving resample of E1b's 80
#   production d_s (section 5.1b); k_frac 1/2 replaces E1b's ceil(5n/8), which cannot fire at large n (section 5.2).
E1C_SEEDS = range(21000, 21704)
E1C = {
    "claimed": DH,
    "looks": tuple(range(64, 705, 64)),
    "sd_pre": 0.3097,
    "bounds": None,
    "k_frac": 0.5,
    "show_beside": True,
}
E1_HP1_T = 2.82  # E1's recorded H-P1 t (the "mirrored bar" signal), beside the BRIDGE read-out
# Addendum 3 (owner ruling #4616): the production tier alone decides deletion; no-encoder is reported beside, never deletes.
SINGLE_TIER_HARM_DELETES = True
NO_ENC_NOTE = "; no-encoder: reported beside, the production tier decides"
ZERO_SD = "degenerate: zero variance"  # A-m4: every d_s equal, so the mde is 0 and no power exists
ARM_OF = {
    "H-P1": "pfrozen",
    "H-P2": "pshuf",
    "H-V3": "vfrozen",
    "H-V1": "replay frozen",
    "H-V2": "replay shuf",
}
N_TICKS = 600
# a seed whose mean reward is below this in every arm it ran is "degenerate"
SENSITIVITY_R = -0.9
# pinned sha256 prefixes of the two e-prop checkpoints
ENC = {"SNN_EPROP_CKPT": "9b471619e39acfa0", "SNN_EPROP_AUDIO_CKPT": "6b32f9eb38beeca3"}
PACKS = ("SNN_PERCEPTION_PACK", "SNN_SHD_PACK")
PCPIN = "SNN_PC_CKPT"  # fifth model asset: a missing path in BOTH tiers; perceive_pc() never runs in a tick
WANT = {"production": ("eprop-snn", True, True), "no-encoder": ("stdlib", False, False)}
ARMS = ("real", "vfrozen", "pfrozen", "vshuf", "pshuf")
E1_RESULTS = os.environ.get(
    "E1B_E1_RESULTS", r"D:\AI-project\lanes\e1-rpe-charge-1019-work\results"
)
SYNTH_SHA = "0" * 40


class Refused(SystemExit):
    """A refusal is an explicit exit with code 3, never an assert: it survives `python -O`."""

    def __init__(self, msg):
        super().__init__(3)
        self.msg = msg


def refuse(tier, stage, why):
    print(f"REFUSED tier={tier} ({stage}): {why}", flush=True)
    raise Refused(why)


def tier_verdict(requested, st):
    """(ok, why): does the brain's reported tier equal the requested one? Pure and brain-free."""
    got = (
        st.get("perception_tier"),
        bool(st.get("perceiving_real")),
        bool(st.get("perceiving_audio")),
    )
    return got == WANT[requested], f"requested={requested} want={WANT[requested]} reported={got}"


def check_optimize(flags=None):
    flags = sys.flags if flags is None else flags
    if flags.optimize != 0:
        refuse("-", "interpreter", f"python -O / PYTHONOPTIMIZE active (optimize={flags.optimize})")


# R3: checked at import by an explicit `if`, so the post-construction check cannot be silently disabled.
if (
    tier_verdict("production", {"perception_tier": "stdlib"})[0]
    or not tier_verdict("no-encoder", {"perception_tier": "stdlib"})[0]
):
    refuse(
        "-", "self-check", "tier_verdict red R3 failed: a stdlib brain must not satisfy production"
    )


def _sha_prefix(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def preflight(requested, env):
    """Env pins and files, BEFORE any brain is built (reachable reds R1/R2). Returns a failure string or None."""
    for k in (*ENC, *PACKS, PCPIN):
        p = env.get(k)
        if not (p and os.path.isabs(p)):
            return f"{k} must be pinned to an ABSOLUTE path, got {p!r}"
    if os.path.exists(env[PCPIN]):
        return f"{PCPIN}={env[PCPIN]} is reachable (must be missing in both tiers)"
    if requested == "production":
        for k, pref in ENC.items():
            if not os.path.isfile(env[k]):
                return f"{k}={env[k]} missing"
            if not _sha_prefix(env[k]).startswith(pref):
                return f"{k} sha256 {_sha_prefix(env[k])} != {pref}"
        for k in PACKS:
            if not os.path.isfile(env[k]):
                return f"{k}={env[k]} missing"
    else:
        for k in (*ENC, *PACKS):
            if os.path.exists(env[k]):
                return f"{k}={env[k]} is reachable"
    return None


def tree_pin(env):
    """The tree the run was told it is on (the runner already refused a moved or dirty tree)."""
    t = {"sha": env.get("E1B_TREE_SHA"), "clean": env.get("E1B_TREE_CLEAN") == "1"}
    sha = t["sha"] or ""
    if not (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha) and t["clean"]):
        refuse("-", "tree", f"tree sha/clean flag not set by the runner: {t}")
    return t


# ------------------------------------------------------------------------------------ statistics
def sd(x):
    if len(x) < 2:
        return float("nan")
    m = sum(x) / len(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - 1))


def ttest(x):
    """One-sample t; +inf for zero variance with a positive mean (E1's frozen rule), -inf with a negative one (A-m4)."""
    if len(x) < 2:
        return float("nan")
    m, s_ = sum(x) / len(x), sd(x)
    if s_ == 0:
        return float("inf") if m > 0 else float("-inf") if m < 0 else 0.0
    return m / (s_ / math.sqrt(len(x)))


def pearson(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return (
        sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)
        if sxx and syy
        else float("nan")
    )


def flips(a, b):
    n = min(len(a), len(b))
    f = [i for i in range(n) if a[i] != b[i]]
    return len(f), (f[0] if f else None)


def _hmask(d, h0):
    return [i for i, t in enumerate(d["v"]["tick"]) if t >= h0]


def mae(d, h0):
    v = d["v"]
    e = [abs(v["pred"][i] - v["r"][i]) for i in _hmask(d, h0) if v["pred"][i] is not None]
    return sum(e) / len(e) if e else float("nan")


def mean_r(d, h0):
    v = d["v"]
    m = _hmask(d, h0)
    return sum(v["r"][i] for i in m) / len(m) if m else float("nan")


def harm(ds):
    return {
        "d": ds,
        "n": len(ds),
        "t": ttest(ds),
        "k": sum(x > 0 for x in ds),
        "sd": sd(ds),
    }


def met(h, n_seeds=N_SEEDS):
    return h["n"] == n_seeds and h["t"] > BAR and h["k"] >= K_MIN


def status(h, claimed, ok=True, n_seeds=N_SEEDS, why="control failed"):
    """MET / VOID / NO VERDICT / NOT MET with the pre-registered power rule. `claimed` is a FROZEN literal."""
    if h["n"] != n_seeds:
        return "NO VERDICT (fewer than the complete seeds)"
    if not ok:
        return f"VOID ({why})"  # neither MET nor NOT MET, and feeds no NULL-RECORD
    if met(h, n_seeds):
        return "MET"
    if h["sd"] == 0:  # A-m4: mde 0 is no power, every d_s is equal
        return f"NOT MET, {ZERO_SD}: INCONCLUSIVE (not powered)"
    mde = MDE_F * h["sd"]
    if mde <= claimed:
        return f"NOT MET, mde {mde:.4f} <= claimed {claimed:.4f}: powered"
    return f"NOT MET, mde {mde:.4f} > claimed {claimed:.4f}: INCONCLUSIVE (underpowered)"


def mde_exact_factor(s, bar=BAR):
    """Informational cross-check of MDE_F: delta/sqrt(S) with power .8 at critical value `bar`, noncentral t by
    numerical integration of chi-density x normal tail (stdlib). The pre-registered MDE_F decides, not this."""
    nu = s - 1

    def pwr(delta):
        top, n = 80.0, 4000
        h = top / n
        tot = 0.0
        for i in range(n + 1):
            v = i * h
            if v == 0.0:
                continue
            f = math.exp(
                (nu / 2 - 1) * math.log(v) - v / 2 - (nu / 2) * math.log(2) - math.lgamma(nu / 2)
            )
            q = 0.5 * math.erfc((bar * math.sqrt(v / nu) - delta) / math.sqrt(2))
            weight = 4 if i % 2 else 2
            tot += f * q * (1 if i == n else weight)
        return tot * h / 3

    lo, hi = 0.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pwr(mid) < 0.8 else (lo, mid)
    return (lo + hi) / 2 / math.sqrt(s)


# distinct-from: instruments/paired_contrast.py -- one fixed-S paired t against one bar + its sign-flip fpr; no looks,
#   no alpha spending, no futility (read; grep for spending/O'Brien/conditional power in src/ finds nothing else)
# distinct-from: instruments/h01_partb_tangential.py, g5_part_d_cells.py, everyday_battery.py, memory_diagnostic.py --
#   the matcher fired on "refuses"/"seeds"/"per segment"; none decides at interim looks (sites read)
# reuses: instruments/pool_selection.t_quantile (lazy import: that module pulls torch, compare is brain-free otherwise)
@functools.cache
@functools.cache  # a pure function of (looks tuple, alpha); sim_guard calls it once per simulated look
def seq_boundaries(looks=LOOKS, alpha=SEQ_ALPHA):
    """Addendum 2: one-sided upper boundaries per look. Lan-DeMets O'Brien-Fleming-type spending
    a(tau) = 2 - 2 Phi(z_{1-alpha/2} / sqrt(tau)), tau = n / cap, solved on the z scale by recursive numerical integration
    of the B-value path (Armitage-McPherson-Rowe; trapezoid grid 0.02; no futility bound inside: futility is non-binding). The t
    boundary keeps each look's nominal p at df = n - 1: t_k = t_{1 - p_k, n_k - 1}."""
    from snn_data_lake.instruments.pool_selection import t_quantile

    nd, cap, h = statistics.NormalDist(), looks[-1], 0.02
    pdf = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)  # noqa: E731 (NormalDist.pdf is ~5x slower)
    sf = lambda x: 0.5 * math.erfc(x / math.sqrt(2))  # noqa: E731
    out, grid, dens, t_prev, spent = [], None, None, 0.0, 0.0
    for n in looks:
        tau = n / cap
        a_k = 2 * sf(
            nd.inv_cdf(1 - alpha / 2) / math.sqrt(tau)
        )  # tail form: 2 - 2 Phi underflows to 0 past ~17 looks
        pi = a_k - spent
        if grid is None:  # look 1: B_1 ~ N(0, tau_1)
            z = -nd.inv_cdf(pi)
        else:
            sd_ = math.sqrt(tau - t_prev)

            def tail(c, sd_=sd_, tau=tau, grid=grid, dens=dens):
                return sum(
                    f * sf((c * math.sqrt(tau) - b) / sd_) for b, f in zip(grid, dens, strict=True)
                )

            lo, hi = 0.0, 10.0
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if tail(mid) > pi else (lo, mid)
            z = (lo + hi) / 2
        top = z * math.sqrt(tau)  # B_k continues below z_k sqrt(tau_k)
        new = [top - h * i for i in range(int((top + 8 * math.sqrt(tau)) / h), -1, -1)]
        if grid is None:
            dens = [h * pdf(b / math.sqrt(tau)) / math.sqrt(tau) for b in new]
        elif n < cap:  # the density past the cap is never read
            dens = [
                h * sum(f * pdf((b - g) / sd_) for g, f in zip(grid, dens, strict=True)) / sd_
                for b in new
            ]
        dens[0] *= 0.5  # trapezoid end weights: `dens` holds probability mass per grid point
        dens[-1] *= 0.5
        grid, t_prev, spent = new, tau, a_k
        p = sf(z)
        out.append({"n": n, "spent": a_k, "z": z, "p": p, "t": t_quantile(1 - p, n - 1)})
    return tuple(out)


# distinct-from: growth/similarity_sized_growth.py, instruments/sleep_plasticity_ab.py -- term overlap only (sites read);
#   so are growth/capacity_certificate.py, faculties/reasoning_curriculum.py, instruments/integration_measure.py,
#   engine/live_brain.py, growth/grower.py, growth/growable.py: none is a look boundary, an mde or a futility rule.
#   graph-exists "t quantile cached per degrees of freedom": NO LEXICAL MATCH
@functools.cache
def _t80(df):
    from snn_data_lake.instruments.pool_selection import t_quantile

    return t_quantile(0.8, df)


def _bound_t(n, looks, alpha, bounds):
    """The look's t boundary: the design's frozen one (addendum 4, H-V1) or the normal-model LD-OBF one."""
    return bounds[looks.index(n)] if bounds else seq_boundaries(looks, alpha)[looks.index(n)]["t"]


def _mde(n, bound_t, sd_):
    return (bound_t + _t80(n - 1)) * sd_ / math.sqrt(n)


def futility_open(n, claimed, looks=LOOKS, alpha=SEQ_ALPHA, *, sd_pre=None, bounds=None):
    """P1 amendment item 2 (re-panel B-N2): a futility stop is taken only at a look before the cap whose mde from the
    PRE-REGISTERED SD is at or below the claimed effect. No pre-registered SD: no futility stop at all."""
    return (
        sd_pre is not None
        and n < looks[-1]
        and _mde(n, _bound_t(n, looks, alpha, bounds), sd_pre) <= claimed
    )


# distinct-from: instruments/paired_contrast.py -- paired t and the exact sign-flip FPR only, no robust location.
#   graph-exists "trimmed mean" / "Hodges-Lehmann median of Walsh averages" / "Wilcoxon signed-rank confidence
#   interval": NO LEXICAL MATCH (2026-10-28); stdlib suffices for a print that gates nothing
def beside(ds, trim=0.10, z=1.959964):
    """E1c section 4, printed beside a look's verdict and gating NOTHING: the sign count k/n, the trimmed mean (`trim`
    cut from each tail), the Hodges-Lehmann shift (median of the Walsh averages) with its ~95% signed-rank interval
    (normal approximation to the signed-rank distribution), and the exact-zero fraction (arms that never diverged)."""
    n = len(ds)
    s = sorted(ds)
    cut = int(trim * n)
    core = s[cut : n - cut] or s
    w = sorted((s[i] + s[j]) / 2 for i in range(n) for j in range(i, n))
    c = max(0, int(n * (n + 1) / 4 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    return {
        "k": sum(x > 0 for x in ds),
        "n": n,
        "trim10": sum(core) / len(core),
        "hl": statistics.median(w),
        "hl_ci": (w[c], w[len(w) - 1 - c]),
        "zeros": sum(x == 0.0 for x in ds) / n,
    }


def seq_decision(
    ds,
    claimed,
    looks=LOOKS,
    alpha=SEQ_ALPHA,
    *,
    sd_pre=None,
    bounds=None,
    k_frac=None,
    show_beside=False,
):
    """Addendum 2: the decision at ONE look on the per-seed differences `ds` (all seeds so far). Refused unless len(ds)
    is a look. harm = t above the look's t boundary AND >= ceil(K_MIN * n / 8) seeds > 0, or >= ceil(k_frac * n) when
    a design passes `k_frac` (E1c: 1/2; E1b's ceil(5n/8) cannot fire at large n on its own shape); futility (non-binding, looks
    before the cap) = conditional power under the frozen claimed effect, observed SD, observed t in place of z, below
    FUTILITY_CP, taken only where `futility_open` (addendum 4); cap = the last look without a crossing. Addendum 3:
    powered = the mde at the STOPPED seed count n with the observed SD, (t_k + t_{.80, n-1}) * SD / sqrt(n), <= claimed.
    A futility or cap stop is never a null by itself (P8 null-scale item 4): powered -> the NULL-RECORD path (one 10x
    re-test before any deletion); else INCONCLUSIVE. `bounds` (addendum 4) replaces the normal-model t boundaries."""
    n = len(ds)
    if n not in looks:
        refuse("any", "sequential", f"a decision at {n} seeds is not a pre-registered look {looks}")
    bs, j, cap, nd = (
        seq_boundaries(looks, alpha),
        looks.index(n),
        looks[-1],
        statistics.NormalDist(),
    )
    h, b, bound_t = harm(ds), bs[looks.index(n)], _bound_t(n, looks, alpha, bounds)
    k_min = math.ceil(K_MIN * n / N_SEEDS) if k_frac is None else math.ceil(k_frac * n)
    crossed = h["t"] > bound_t and h["k"] >= k_min
    mde = _mde(n, bound_t, h["sd"])
    cp = None
    if not crossed and n < cap:
        tau = n / cap
        drift = claimed / h["sd"] * math.sqrt(cap) if h["sd"] > 0 else math.inf
        cp = 1 - nd.cdf(
            (bs[-1]["z"] - h["t"] * math.sqrt(tau) - drift * (1 - tau)) / math.sqrt(1 - tau)
        )
    is_open = futility_open(n, claimed, looks, alpha, sd_pre=sd_pre, bounds=bounds)
    if crossed:
        decision = "harm"
    elif n == cap:
        decision = "cap"
    else:
        decision = "futility" if is_open and cp < FUTILITY_CP else "continue"
    where = f"look {j + 1} of {len(looks)}, n={n}"
    powered = h["sd"] > 0 and mde <= claimed  # A-m4: a zero observed SD gives mde 0, not power
    power = f"mde at n={n} {mde:.4f} {'<=' if mde <= claimed else '>'} claimed {claimed:.4f}: " + (
        "powered: a null, NULL-RECORD path (one 10x re-test before any deletion)"
        if powered
        else f"{ZERO_SD}, not powered: INCONCLUSIVE (P8 null-scale item 4): no NULL-RECORD, no deletion"
        if h["sd"] == 0
        else "INCONCLUSIVE when underpowered (P8 null-scale item 4): no NULL-RECORD, no deletion"
    )
    status_ = {
        "harm": f"MET at look {j + 1} ({n} seeds)",
        "continue": f"CONTINUE at {where} (no verdict): next look n={looks[min(j + 1, len(looks) - 1)]}",
        "futility": f"NOT MET, stopped for futility at {where} (conditional power {cp or 0:.3f} < {FUTILITY_CP}), {power}",
        "cap": f"NOT MET at the cap ({where}), {power}",
    }[decision]
    bes = beside(ds) if show_beside else None
    if bes:  # E1c section 4: printed beside the verdict; it changes no field the decision reads
        lo, hi = bes["hl_ci"]
        status_ += (
            f" | beside (no bar): k {bes['k']}/{bes['n']}, trim10 {bes['trim10']:+.4f},"
            f" HL {bes['hl']:+.4f} [{lo:+.4f}, {hi:+.4f}], zeros {bes['zeros']:.3f}"
        )
    return {
        "beside": bes,
        "look": j + 1,
        "n_looks": len(looks),
        "n": n,
        "t": h["t"],
        "k": h["k"],
        "k_min": k_min,
        "bound_t": bound_t,
        "bound_z": b["z"],
        "cp": cp,
        "futility_open": is_open,
        "mde": mde,
        "powered": powered,
        "decision": decision,
        "status": status_,
    }


# reuses: seq_decision (one look's decision) -- this walks it over the look prefixes; graph-exists: NO LEXICAL MATCH
def seq_record(ds, claimed, looks=LOOKS, alpha=SEQ_ALPHA, **design):
    """Addendum 2's decision of record: the FIRST stop (harm, futility or the cap) over the look prefixes of the
    seed-ordered `ds`; CONTINUE at len(ds) when no prefix has stopped. Refused unless len(ds) is a look."""
    if len(ds) not in looks:
        refuse(
            "any",
            "sequential",
            f"a decision at {len(ds)} seeds is not a pre-registered look {looks}",
        )
    for n in looks[: looks.index(len(ds)) + 1]:
        q = seq_decision(ds[:n], claimed, looks, alpha, **design)
        if q["decision"] != "continue":
            break
    return q


# reuses: seq_decision (the same rule the campaign reads); graph-exists "zero-preserving resample of a measured
#   per-seed shape": NO LEXICAL MATCH -- the design measurement it replaces lives outside the tree (derive_e1c.py)
# distinct-from: instruments/pool_selection.py -- step_tune reads per-candidate D through a readout; no resampling
# distinct-from: faculties/recursive_tom.py -- t082_contrasts decides one campaign's conjuncts on measured readings
# distinct-from: instruments/attention_schema.py -- g4_rule is one campaign's decode rule on measured readings
# distinct-from: instruments/replay_hydration.py -- seed_comparisons compares recorded arms seed by seed
# distinct-from: instruments/carried_state_campaign.py -- oracle_check is a per-seed accuracy contrast row
#   (none resamples a measured per-seed shape through a sequential stopping rule to estimate its stop rates)
def sim_guard(
    ds, design=None, *, campaigns=2000, name="e1c", scale=1.0, effect=0.0, zero_frac=None
):
    """E1c R5 (section 5.3) and the validity read's power recompute: resample a MEASURED per-seed shape `ds` through
    `seq_decision` under `design` (default E1C) and return the rate of each stop: harm, futility-/cap- x powered/under.
    Zero-preserving (section 5.1b): exact zeros are seeds whose arms never diverged and stay 0; the diverged seeds are
    centred so the zero-kept null has overall mean 0, and `effect` is carried by the diverged seeds alone, so the
    overall mean shift equals `effect`. `zero_frac` overrides the measured zero fraction (the validity read past
    12.5%); `scale` stretches the diverged spread (the 80% upper SD). Seeded by `name`, so a run reproduces."""
    design = dict(design or E1C, show_beside=False)
    claimed, looks = design.pop("claimed"), design.pop("looks")
    nz = [x for x in ds if x != 0.0] or [0.0]
    zf = (len(ds) - len(nz)) / len(ds) if zero_frac is None else zero_frac
    mu = sum(nz) / len(nz)
    base = [x - mu for x in nz]
    shift = effect / (1 - zf) if zf < 1 else 0.0
    rng, tally = random.Random(f"{name}/{scale}/{effect}/{zf}"), {}
    for _ in range(campaigns):
        d = []
        for n in looks:
            while len(d) < n:
                d.append(0.0 if rng.random() < zf else rng.choice(base) * scale + shift)
            q = seq_decision(d, claimed, looks, **design)
            if q["decision"] != "continue":
                break
        kind = q["decision"]
        key = kind if kind == "harm" else f"{kind}-{'powered' if q['powered'] else 'under'}"
        tally[key] = tally.get(key, 0) + 1
    return {k: v / campaigns for k, v in sorted(tally.items())}


# ---------------------------------------------------------------------------- arms (brain-bound)
def _perm(seq, seed):
    idx = list(range(len(seq)))
    random.Random(seed * 1009 + 17).shuffle(idx)
    return [seq[i] for i in idx]


class ArmPatch:
    """E1's patches, restored on exit. Class-level wrappers that act ONLY on the bound arm's own
    instances (identity filter); foreign ValenceLearner calls are counted and passed through. Every wrapper records, then
    re-raises: a swallowed exception would make a broken patch look like a perfect neuter."""

    def __init__(self, arm, perm=None):
        self.arm, self.perm = arm, perm
        self.v = {
            "foreign_valence": 0,
            "foreign_learn": 0,
            "calls": 0,
            "pattern": [],
            "r": [],
            "pred": [],
            "rpe": [],
            "tick": [],
            "r_sent": [],
        }
        self.p = {"calls": 0, "rpe_entry": [], "rpe_used": [], "tick": [], "plan_w_before": []}
        self.exc = []
        self.tick = -1
        self.pc_calls = 0
        self.target = None
        self._saved = []
        self._pending = None  # (tick, valence) of the last prediction made outside learn()
        self._in_learn = False

    def bind(self, arm_obj):
        self.target = arm_obj
        if self.arm == "pfrozen":
            arm_obj._plan_lr = 0.0

    def _mine(self, obj, attr):
        return self.target is not None and obj is getattr(self.target, attr, None)

    def __enter__(self):
        from snn_data_lake.engine.autonomous_loop import AutonomousArm
        from snn_data_lake.faculties.prefrontal_executive import PrefrontalExecutive
        from snn_data_lake.plasticity.valence_stdp import ValenceLearner

        ov, ol, op = ValenceLearner.valence, ValenceLearner.learn, AutonomousArm._pfc_bias
        self._saved = [
            (ValenceLearner, "valence", ov),
            (ValenceLearner, "learn", ol),
            (AutonomousArm, "_pfc_bias", op),
        ]
        V, P, me = self.v, self.p, self

        def w_valence(slf, window):
            if not me._mine(slf, "_valence"):
                V["foreign_valence"] += 1
                return ov(slf, window)
            try:
                out = ov(slf, window)
                if not me._in_learn:  # learn()/dopamine() call valence internally: not the pred
                    me._pending = (me.tick, float(out))
                return out
            except Exception as e:
                me.exc.append({"tick": me.tick, "where": "valence", "exc": repr(e)})
                raise

        def w_learn(slf, window, reward, lr_scale=1.0):
            if not me._mine(slf, "_valence"):
                V["foreign_learn"] += 1
                return ol(slf, window, reward, lr_scale)
            try:
                pat = list(window)
                r_sent = reward
                me._in_learn = True
                try:
                    if me.arm == "vfrozen":
                        out = slf.dopamine(window, reward)
                    elif me.arm == "vshuf":
                        r_sent = me.perm[V["calls"] % len(me.perm)]
                        out = ol(slf, window, r_sent, lr_scale)
                    else:
                        out = ol(slf, window, reward, lr_scale)
                finally:
                    me._in_learn = False
                pend, me._pending = me._pending, None
                V["tick"].append(me.tick)
                V["pattern"].append(pat)
                V["r"].append(float(reward))
                V["r_sent"].append(float(r_sent))
                V["pred"].append(pend[1] if pend else None)
                V["rpe"].append(float(out))
                V["calls"] += 1  # AFTER the wrapped body returned
                return out
            except Exception as e:
                me.exc.append({"tick": me.tick, "where": "learn", "exc": repr(e)})
                raise

        def w_pfc_bias(slf, unit, pattern=None, qa=None):
            if slf is not me.target:
                return op(slf, unit, pattern, qa)
            try:
                entry = float(slf._rpe or 0.0)
                used = entry
                pw = float(slf._plan_w)
                saved = slf._rpe
                if me.arm == "pshuf":
                    used = float(me.perm[P["calls"] % len(me.perm)])
                    slf._rpe = used
                try:
                    out = op(slf, unit, pattern, qa)
                finally:
                    if me.arm == "pshuf":
                        slf._rpe = saved
                P["tick"].append(me.tick)
                P["rpe_entry"].append(entry)
                P["rpe_used"].append(used)
                P["plan_w_before"].append(pw)
                P["calls"] += 1  # AFTER the wrapped body returned
                return out
            except Exception as e:
                me.exc.append({"tick": me.tick, "where": "_pfc_bias", "exc": repr(e)})
                raise

        ValenceLearner.valence, ValenceLearner.learn, AutonomousArm._pfc_bias = (
            w_valence,
            w_learn,
            w_pfc_bias,
        )
        if self.arm == "pc":
            self._saved.append(
                (PrefrontalExecutive, "_decide_spiking", PrefrontalExecutive._decide_spiking)
            )

            def stub(slf, evidence):
                me.pc_calls += 1
                return 0, 1

            PrefrontalExecutive._decide_spiking = stub
        return self

    def __exit__(self, *exc):
        for cls, name, fn in reversed(self._saved):
            setattr(cls, name, fn)
        return False


def install_counters(b, exc):
    """Instance-scoped encoder-use counters (record, then re-raise). A missing encoder (None) is not wrapped: its
    count stays 0, which is what the no-encoder tier requires and what makes a wrong attribute path fail closed."""
    ec = {"vision": 0, "audio": 0, "pc": 0}

    def count(obj, name, key):
        if obj is None or not hasattr(obj, name):
            return
        orig = getattr(obj, name)

        def w(*x, **k):
            ec[key] += 1
            try:
                return orig(*x, **k)
            except BaseException as e:
                exc.append({"where": f"{key}.{name}", "exc": repr(e)})
                raise

        setattr(obj, name, w)

    for n in ("encode", "classify"):
        count(getattr(b, "_perception_enc", None), n, "vision")
        count(getattr(b, "_audio_enc", None), n, "audio")
    count(b, "perceive_pc", "pc")
    return ec


def _out_problem(out):
    """Why `out` must not be written, or None: a directory/junction, or a link (writing it writes through to its target)."""
    if os.path.isdir(out) or os.path.isjunction(out):
        return f"{out} is a directory or junction"
    if os.path.lexists(out) and (os.path.islink(out) or os.stat(out).st_nlink > 1):
        return f"{out} is a link: writing it would write through to another file"
    return None


def run_arm(arm, seed, ticks, tier, home, out, series=None, env=None):
    """ONE arm in THIS process: preflight -> construct -> tier check -> ticks -> JSON. Refusals raise `Refused`
    (exit 3) before any result file is written; the env, path and series refusals fire before any brain exists."""
    env = os.environ if env is None else env
    check_optimize()
    pf = preflight(tier, env)
    if pf:
        refuse(tier, "preflight", pf)
    tree = tree_pin(env)
    from snn_data_lake.instruments.sleep_plasticity_ab import (
        guard_out,
    )  # the repo's pre-mutation path guard
    from snn_data_lake.substrate import home as H

    out_dir = os.path.dirname(os.path.abspath(out))
    if str(home).startswith("\\\\"):
        refuse(tier, "home", f"{home!r} is a device-namespace or share spelling")
    try:
        guard_out(home)
        guard_out(out_dir)
    except H.LiveHomeRefused as e:
        refuse(tier, "home", str(e))
    if (why := _out_problem(out)) is not None:
        refuse(tier, "out", why)
    for k in ("NORD_PERSIST", "NORD_SELFMOD", "NORD_ESTOP_SELF_CLEAR"):
        if os.environ.get(k):
            refuse(tier, "env", f"{k} must be unset")
    perm = None
    if arm in ("vshuf", "pshuf"):
        if not series:
            refuse(tier, "series", "--series required for shuffle arms")
        with open(series, encoding="utf-8") as f:
            real = json.load(f)
        if real["arm"] != "real" or real["seed"] != seed:
            refuse(tier, "series", "series must be the same seed's real arm")
        rt = real.get("tier") or {}
        if (rt.get("requested"), rt.get("reported")) != (tier, WANT[tier][0]):
            refuse(tier, "series", f"series is a {rt.get('requested')} run, not {tier}")
        if rt.get("tree_sha") != tree["sha"]:
            refuse(tier, "series", "series was recorded on another tree sha")
        perm = _perm(real["v"]["r"] if arm == "vshuf" else real["p"]["rpe_entry"], seed)
    os.makedirs(home, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)  # a late FileNotFoundError would throw away the whole run
    try:  # THE shared guard (P2): equality and containment both ways, aliases, shares, \\?\ spellings
        os.environ["NORD_HOME"] = H.assert_not_live_home(home)
        os.environ["SNDL_LIVE_HOME_GUARD"] = "1"  # arms home()'s own tripwire in this process
        H.require_isolated_home()
        H.assert_not_live_home(out_dir)
    except H.LiveHomeRefused as e:
        refuse(tier, "home", str(e))
    os.environ.setdefault("NORD_GPU", "1")
    for k in (*ENC, *PACKS, PCPIN):
        os.environ[k] = env[k]
    from snn_data_lake.engine.autonomous_loop import AutonomousArm
    from snn_data_lake.engine.brain import make_brain

    with ArmPatch(arm, perm) as ap:
        t0 = time.perf_counter()
        b = make_brain(seed=seed, persist=False)
        construct_s = time.perf_counter() - t0
        print(f"construct_s={construct_s:.1f} arm={arm} seed={seed}", flush=True)
        st = {
            "perception_tier": b._perception_tier,
            "perceiving_real": b._perceiving_real,
            "perceiving_audio": b._perceiving_audio,
        }
        ok, why = tier_verdict(tier, st)
        print(f"TIER {why} ok={ok}", flush=True)
        if not ok:
            refuse(tier, "reported", why)
        ec = install_counters(b, ap.exc)
        pt = {"vision": 0, "audio": 0}
        arm_obj = b.brain.arm
        if (
            not isinstance(arm_obj, AutonomousArm)
            or getattr(b, "_persist", None)
            or not arm_obj._dopa_on
        ):
            refuse(
                tier,
                "brain",
                "arm is not an AutonomousArm with the dopamine block active and persist off",
            )
        ap.bind(arm_obj)
        s = {
            k: []
            for k in (
                "tick",
                "action",
                "da_level",
                "rpe",
                "plan_w",
                "plan_elig",
                "plan_base",
                "phase",
            )
        }
        errs = 0
        for i in range(ticks):
            ap.tick = i
            try:
                ph = b.tick()
            except Exception as e:
                errs += 1
                ph = "err"
                print(f"tick {i} err: {type(e).__name__}: {str(e)[:160]}", flush=True)
            res = b._last_res or {}
            s["tick"].append(i)
            s["phase"].append(ph)
            s["action"].append(res.get("action"))
            s["da_level"].append(float(getattr(arm_obj, "_da_level", 0.0) or 0.0))
            s["rpe"].append(float(getattr(arm_obj, "_rpe", 0.0) or 0.0))
            s["plan_w"].append(float(arm_obj._plan_w))
            s["plan_elig"].append(float(arm_obj._plan_elig))
            s["plan_base"].append(float(arm_obj._plan_base))
            pt["vision"] += 1 if getattr(b, "_real_percept", None) else 0
            pt["audio"] += 1 if getattr(b, "_audio_percept", None) else 0
        v = ap.v
        rec = {
            "arm": arm,
            "seed": seed,
            "ticks": ticks,
            "errs": errs,
            "construct_s": round(construct_s, 1),
            "run_s": round(time.perf_counter() - t0, 1),
            "swallowed_exceptions": len(ap.exc),
            "exceptions": ap.exc,
            "learn_calls": v["calls"],
            "foreign_valence": v["foreign_valence"],
            "foreign_learn": v["foreign_learn"],
            "pfc_bias_calls": ap.p["calls"],
            "pc_stub_calls": ap.pc_calls,
            "pattern_len_ok": len(v["pattern"]) == v["calls"],
            "shuffle_wrapped": bool(perm)
            and (v["calls"] if arm == "vshuf" else ap.p["calls"]) > len(perm),
            "perm_len": len(perm) if perm else None,
            "plan_lr_final": float(arm_obj._plan_lr),
            "persist": bool(getattr(b, "_persist", None)),
            "nord_persist_env": os.environ.get("NORD_PERSIST"),
            "tier": {
                "requested": tier,
                "reported": st["perception_tier"],
                "perceiving_real": st["perceiving_real"],
                "perceiving_audio": st["perceiving_audio"],
                "state_api": None,
                "pins": {k: env[k] for k in (*ENC, *PACKS, PCPIN)},
                "sha": {k: _sha_prefix(env[k]) for k in ENC} if tier == "production" else {},
                "pc_cortex_tier": getattr(b, "_pc_cortex_tier", None),
                "optimize": sys.flags.optimize,
                "tree_sha": tree["sha"],
                "tree_clean": tree["clean"],
                "encoder_calls": dict(ec),
                "percept_ticks": dict(pt),
            },
            "v": v,
            "p": ap.p,
            "series": s,
        }
        rec = json.loads(
            json.dumps(rec)
        )  # state() runs only AFTER the record is complete and copied
        try:
            rec["tier"]["state_api"] = b.state()["perception_tier"]
        except Exception as e:  # recorded, not swallowed: compare VOIDs on a mismatch
            rec["tier"]["state_api"] = f"state() raised {e!r}"
    try:  # the checks above ran minutes ago: repeat them at write time, then replace the entry, never write through it
        H.assert_not_live_home(out_dir)
    except H.LiveHomeRefused as e:
        refuse(tier, "home", str(e))
    if (why := _out_problem(out)) is not None:
        refuse(tier, "out", why)
    tmp = f"{out}.{os.getpid()}.tmp"
    with open(tmp, "x", encoding="utf-8") as f:
        json.dump(rec, f)
    os.replace(tmp, out)
    print(
        f"DONE arm={arm} seed={seed} errs={errs} learn_calls={v['calls']} pfc_calls={ap.p['calls']} "
        f"exc={len(ap.exc)} len_ok={rec['pattern_len_ok']} ticks={ticks} run_s={rec['run_s']} tier={tier}/{st['perception_tier']}",
        flush=True,
    )
    return rec


# -------------------------------------------------------------------------- replay (brain-free)
def score(pats, rs, ticks, h0, mode, seed):
    """Mean |valence_t(pattern_t) - r_t| over H, prediction BEFORE learning on call t, scored vs the TRUE reward."""
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner

    vl = ValenceLearner()
    rl = _perm(rs, seed) if mode == "shuf" else rs
    err = []
    for k, (p, r) in enumerate(zip(pats, rs)):
        pred = vl.valence(p)
        if ticks[k] >= h0:
            err.append(abs(pred - r))
        if mode == "learn":
            vl.learn(p, r)
        elif mode == "shuf":
            vl.learn(p, rl[k])
    return (sum(err) / len(err) if err else None), len(err)


@functools.cache
def _planted(kind, seeds):
    """Planted streams (600 ticks, two patterns). learn: fixed +1/-1 per pattern (learning helps). harm: reward
    ALTERNATES sign on each occurrence of a pattern (learning hurts). null: reward 0 everywhere (nothing to learn)."""
    from snn_data_lake.plasticity.valence_stdp import _pattern

    rows = {}
    for seed in seeds:
        rng = random.Random(seed if kind != "harm" else 1000 + seed)
        pa, pb = _pattern(1), _pattern(2)
        sgn = [1.0, 1.0]
        pats, rs = [], []
        for _ in range(600):
            i = 0 if rng.random() < 0.5 else 1
            pats.append(pa if i == 0 else pb)
            if kind == "learn":
                rs.append(1.0 if i == 0 else -1.0)
            elif kind == "harm":
                rs.append(sgn[i])
                sgn[i] = -sgn[i]
            else:
                rs.append(0.0)
        rows[str(seed)] = {
            m: score(pats, rs, list(range(600)), 300, m, seed)[0]
            for m in ("learn", "frozen", "shuf")
        }
    return rows


def planted_rows(kind, seeds=PSEEDS):
    """The planted rows of `_planted`, cached (the streams are fixed) and handed out as a copy."""
    return copy.deepcopy(_planted(kind, tuple(seeds)))


def control_pass(rows):
    """E1's planted-learning control: t(frozen-learn) and t(shuf-learn) over the planted seeds both > BAR."""
    ta = ttest([r["frozen"] - r["learn"] for r in rows.values()])
    tb = ttest([r["shuf"] - r["learn"] for r in rows.values()])
    return ta, tb, bool(ta > BAR and tb > BAR)


def harm_control_pass(rows):
    """(t, ok): the planted-HARM control, t(learn - frozen) over the planted seeds against BAR."""
    t = ttest([r["learn"] - r["frozen"] for r in rows.values()]) if rows else float("nan")
    return t, bool(t > BAR)


# distinct-from: instruments/carried_state_report.py -- its digest_of is the same canonical-JSON sha256, but importing
#   it pulls torch and the reservoir into the brain-free compare; a run's series is plain JSON already (stdlib suffices)
def replay_binding(run):
    """Re-panel A-M2: what ties a replay row to the `real` run it replayed (N / n_calls alone tie nothing): the run's
    recorded tree sha and the sha256 of its replayed (pattern, r, tick) series."""
    v = run["v"]
    text = json.dumps([v["pattern"], v["r"], v["tick"]], allow_nan=False)
    return {
        "tree_sha": (run.get("tier") or {}).get("tree_sha"),
        "digest": hashlib.sha256(text.encode()).hexdigest(),
    }


def replay(rows_by_seed_runs, n=None):
    """e2v2 dict from {seed: real-run json}: fresh ValenceLearner per mode, plus both planted controls."""
    out = {"real": {}, "planted": planted_rows("learn"), "planted_harm": planted_rows("harm")}
    for seed, d in rows_by_seed_runs.items():
        nn = n or d["ticks"]
        v = d["v"]
        row = {"n_calls": len(v["r"]), "N": nn, **replay_binding(d)}
        for mode in ("learn", "frozen", "shuf"):
            row[mode], row["n_scored"] = score(v["pattern"], v["r"], v["tick"], nn // 2, mode, seed)
        out["real"][str(seed)] = row
    return out


# ------------------------------------------------------------------------------- loading / logs
def load_run(results_dir, tier, tag):
    p = os.path.join(results_dir, tag + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    t = d.get("tier") or {}
    if (t.get("requested"), t.get("reported")) != (
        tier,
        WANT[tier][0],
    ):  # a result of the other tier never enters
        raise SystemExit(f"TIER MISMATCH in {p}: {t}")
    return d


def load_runs(results_dir, tier, seeds=SEEDS):
    runs = {}
    for arm in ARMS:
        for s in seeds:
            d = load_run(results_dir, tier, f"{arm}_{s}")
            if d:
                runs[(arm, s)] = d
    s0 = list(seeds)[0]
    for key, tag in (
        (("real_r2", s0), f"real_{s0}_r2"),
        (("pc", s0), f"pc_{s0}"),
        (("pc", s0 + 1), f"pc_{s0 + 1}"),
    ):
        d = load_run(results_dir, tier, tag)
        if d:
            runs[key] = d
    return runs


FIXTURE_WANT = (
    ("harm", "H-P1 MET"),
    ("noharm", "H-P1 CONTINUE at look 1"),
    ("void", "H-V1 VOID (control failed)"),
)


def fixture_ok(lines):
    """The three fixture verdicts of the frozen section 4: harm -> MET, failed control -> VOID; no-harm -> NOT MET there,
    CONTINUE at look 1 (no verdict) under addendum 2 (re-panel C R2-M1: a continuing look is never printed NOT MET)."""
    return len(lines) == len(FIXTURE_WANT) and all(
        ln.startswith(f"fixture {kind} ") and want in ln
        for (kind, want), ln in zip(FIXTURE_WANT, lines, strict=True)
    )


def logs_gate(logs_dir):
    """tier_red.log must hold a refused red for EACH tier ('rc=3 result_file_exists=False'); fixture.log must end with
    the three expected fixture verdicts AND they must be what the code prints now."""
    try:
        with open(os.path.join(logs_dir, "tier_red.log"), encoding="utf-8") as f:
            red = f.read().splitlines()
        with open(os.path.join(logs_dir, "fixture.log"), encoding="utf-8") as f:
            fx = [ln for ln in f.read().splitlines() if ln.startswith("fixture ")]
    except OSError:
        return False
    refused = [ln for ln in red if "rc=3 result_file_exists=False" in ln]
    both = all(any(f"tier={t}" in ln for ln in refused) for t in WANT)
    now = fixture_lines()
    return both and fixture_ok(fx[-len(FIXTURE_WANT) :]) and fx[-len(FIXTURE_WANT) :] == now


# --------------------------------------------------------------------------- run-level validity
def run_problems(arm, key, d, tier):
    """Per-run harness validity (E1's set + Revision 2's tier block)."""
    tag = f"{arm}_{key}"
    bad = []
    if d["swallowed_exceptions"] != 0:
        bad.append(f"{tag}: swallowed={d['swallowed_exceptions']}")
    if not d["pattern_len_ok"]:
        bad.append(f"{tag}: pattern series length != learn calls")
    if d["errs"]:
        bad.append(f"{tag}: tick errs={d['errs']}")
    if d["ticks"] != N_TICKS:
        bad.append(f"{tag}: ticks={d['ticks']} != {N_TICKS} (N is fixed by the pre-registration)")
    if arm in ("vfrozen", "vshuf", "real") and d["learn_calls"] == 0:
        bad.append(f"{tag}: learn_calls==0")
    if arm in ("pfrozen", "pshuf") and d["pfc_bias_calls"] == 0:
        bad.append(f"{tag}: pfc_bias_calls==0")
    if arm == "pfrozen" and d["plan_lr_final"] != 0.0:
        bad.append(f"{tag}: plan_lr_final={d['plan_lr_final']}")
    if arm == "pc" and d["pc_stub_calls"] == 0:
        bad.append(f"{tag}: stub calls==0")
    if any(x is None for x in d["v"]["pred"]):
        bad.append(f"{tag}: pred is None on a recorded call")
    elif arm == "real" and any(
        abs(pr - (r - rp)) >= 1e-9 for pr, r, rp in zip(d["v"]["pred"], d["v"]["r"], d["v"]["rpe"])
    ):
        bad.append(f"{tag}: pred != r - rpe on some call (E-V1 VOID)")
    if d["persist"] or d["nord_persist_env"]:
        bad.append(f"{tag}: persist on")
    t = d["tier"]
    if (t["requested"], t["reported"]) != (tier, WANT[tier][0]):
        bad.append(f"{tag}: tier {t['requested']}/{t['reported']} is not {tier}")
    if t["state_api"] != t["reported"]:
        bad.append(f"{tag}: state() tier {t['state_api']!r} != attribute {t['reported']!r}")
    if (t["perceiving_real"], t["perceiving_audio"]) != WANT[tier][1:]:
        bad.append(
            f"{tag}: perceiving flags {t['perceiving_real']}/{t['perceiving_audio']} do not match tier {tier}"
        )
    if tier == "production" and t["sha"] != ENC:
        bad.append(f"{tag}: encoder sha256 prefixes {t['sha']} != pinned")
    if t["optimize"] != 0:
        bad.append(f"{tag}: optimize={t['optimize']}")
    if t["pc_cortex_tier"] != "stdlib" or t["encoder_calls"]["pc"]:
        bad.append(f"{tag}: pc cortex active")
    e, p_ = t["encoder_calls"], t["percept_ticks"]
    if tier == "production" and not (
        e["vision"] > 0 and e["audio"] > 0 and p_["vision"] > 0 and p_["audio"] > 0
    ):
        bad.append(f"{tag}: production run with zero encoder use {e}/{p_}")
    if tier == "no-encoder" and (e["vision"] or e["audio"] or p_["vision"] or p_["audio"]):
        bad.append(f"{tag}: no-encoder run with encoder use {e}/{p_}")
    return bad


def analyse(tier, runs, e2, *, seeds=SEEDS, ep1_min=EP1_MIN, logs_pass=False):
    """All statistics for ONE tier, from run JSON only. `logs_pass` is the tier_red/fixture gate (`logs_gate(dir)`)."""
    seeds = list(seeds)
    s0, ns = seeds[0], len(seeds)
    if ns not in CAMPAIGN_LOOKS:
        refuse(
            tier,
            "compare",
            f"{ns} seeds is not a pre-registered look {CAMPAIGN_LOOKS} (addenda 2, 4, 5, 7, 9)",
        )
    # each endpoint reads the seed block its pre-registration names (re-panel A-B1): the secondaries and the
    # dispositions the look-1 block, H-P1 and H-V1 every look up to their own caps
    n_ticks = next((d["ticks"] for d in runs.values()), None)
    h0 = n_ticks // 2 if n_ticks else 0
    rows_all = e2["real"] if e2 else {}
    d_p1 = {
        s: mean_r(runs[("pfrozen", s)], h0) - mean_r(runs[("real", s)], h0)
        for s in seeds
        if ("real", s) in runs and ("pfrozen", s) in runs
    }
    d_v1 = {
        s: rows_all[str(s)]["learn"] - rows_all[str(s)]["frozen"]
        for s in seeds
        if str(s) in rows_all
    }
    # each primary carried from its own stop of record, else its cap (A-M4; addenda 3-9)
    n_exp = {"H-P1": _n_owed("H-P1", seeds, d_p1), "H-V1": _n_owed("H-V1", seeds, d_v1)}
    block, p1_seeds, v1_seeds = seeds[:N_SEEDS], seeds[: n_exp["H-P1"]], seeds[: n_exp["H-V1"]]
    a = {"tier": tier, "seeds": seeds, "n_owed": n_exp}
    a.update(N=n_ticks, H0=h0)
    bad, val, pending, ctl_bad = (
        [],
        [],
        [],
        [],
    )  # pending = runs/controls not yet there (a partial campaign), never VOID
    for (arm, s), d in sorted(runs.items()):
        bad += run_problems("real" if arm == "real_r2" else arm, s, d, tier)
        val.append(
            f"{arm}_{s}: foreign valence calls={d['foreign_valence']} foreign learn calls={d['foreign_learn']}"
        )
    shas = {d["tier"]["tree_sha"] for d in runs.values()}
    a["shas"] = sorted(map(str, shas))
    val.append(f"tree sha(s) in this tier: {a['shas']}")
    if len(shas) != 1 or None in shas:
        bad.append("runs of this tier do not share one recorded tree sha")
    if any(not d["tier"]["tree_clean"] for d in runs.values()):
        bad.append("a run recorded tree_clean = false")
    rep = runs.get(("real_r2", s0))
    if rep and ("real", s0) in runs:
        b0 = runs[("real", s0)]
        rf = flips(b0["series"]["action"], rep["series"]["action"])
        same = (
            all(b0["v"][k] == rep["v"][k] for k in ("pattern", "r", "pred", "rpe"))
            and b0["series"] == rep["series"]
        )
        val.append(
            f"repeat arm (real seed {s0} again): action flips {rf[0]} first={rf[1]}; recorded series identical: {same}"
        )
        if rf[0] != 0 or not same:
            bad.append(f"repeat arm differs from real seed {s0}")
        floor = rf[0]
    else:
        val.append("repeat arm MISSING")
        pending.append("repeat arm missing")
        floor = 0
    for s in seeds[:2]:
        if ("pc", s) in runs and ("real", s) in runs:
            f = flips(runs[("real", s)]["series"]["action"], runs[("pc", s)]["series"]["action"])
            val.append(f"pc seed {s}: action flips {f[0]} (first {f[1]})")
            if f[0] <= 0:
                bad.append(f"pc seed {s} flips == 0")
        else:
            pending.append(f"pc seed {s} missing")
    for arm in ARMS:  # later looks add `real`, and `pfrozen` up to H-P1's stop (addenda 2, 5-7, 9)
        for s in seeds if arm == "real" else p1_seeds if arm == "pfrozen" else block:
            if (arm, s) not in runs:
                pending.append(f"{arm}_{s} missing")
    for arm in ("vshuf", "pshuf"):
        w = [s for s in seeds if (arm, s) in runs and runs[(arm, s)]["shuffle_wrapped"]]
        if w:
            val.append(f"{arm}: call count exceeded real series length (modulo wrap) on seeds {w}")
    pc_ok = ph_ok = None
    if e2:
        ta, tb, pc_ok = control_pass(e2["planted"])
        val.append(
            f"E-V2 planted-learning control: t(frozen-learn)={ta:.2f} t(shuf-learn)={tb:.2f} -> {'PASS' if pc_ok else 'FAIL'}"
        )
        if not pc_ok:
            ctl_bad.append("E-V2 planted control failed")
        for k, r in e2["real"].items():
            run = runs.get(("real", int(k)))
            n_run = len(run["v"]["r"]) if run else None
            if r.get("N", N_TICKS) != N_TICKS or r.get("n_calls", n_run) != n_run and run:
                bad.append(
                    f"replay row seed {k}: N/n_calls do not match the current real run (stale or mis-sized)"
                )
            elif run and any(r.get(f) != b for f, b in replay_binding(run).items()):
                bad.append(
                    f"replay row seed {k}: tree sha / series digest do not bind it to the current real run"
                )
        if not e2.get("planted_harm"):
            ctl_bad.append("planted-harm control absent from the replay record")
        else:
            th, ph_ok = harm_control_pass(e2["planted_harm"])
            val.append(
                f"planted-HARM control: t(learn-frozen)={th:.2f} -> {'PASS' if ph_ok else 'FAIL'}"
            )
            if not ph_ok:
                ctl_bad.append("planted-harm control failed")
    else:
        pending.append("e2v2.json missing")
    if not logs_pass:
        bad.append("logs/tier_red.log or logs/fixture.log missing or lacking the expected lines")
    bad += ctl_bad
    valid = not bad
    void_why = "control failed" if ctl_bad and not set(bad) - set(ctl_bad) else "harness validity"
    ctl_ok = bool(pc_ok) and bool(ph_ok)
    a.update(bad=bad, val=val, valid=valid, pc_ok=pc_ok, ph_ok=ph_ok, ctl_ok=ctl_ok, floor=floor)
    a["pending"] = pending

    have = lambda *arms, among=seeds: [s for s in among if all((x, s) in runs for x in arms)]  # noqa: E731
    mr_ = lambda arm, s: mean_r(runs[(arm, s)], h0)  # noqa: E731
    mae_ = lambda arm, s: mae(runs[(arm, s)], h0)  # noqa: E731
    d1 = [
        mae_("vfrozen", s) - mae_("real", s) for s in have("real", "vfrozen", "vshuf", among=block)
    ]
    d2 = [mae_("vshuf", s) - mae_("real", s) for s in have("real", "vfrozen", "vshuf", among=block)]
    a["ev1"] = {
        "t": [ttest(d1), ttest(d2)],
        "k": [sum(x > 0 for x in d1), sum(x > 0 for x in d2)],
        "full": len(d1) == N_SEEDS,
        "d": [d1, d2],
    }
    a["ev1"]["met"] = (
        a["ev1"]["full"]
        and a["ev1"]["t"][0] > BAR
        and a["ev1"]["t"][1] > BAR
        and min(a["ev1"]["k"]) >= K_MIN
    )
    rows = {str(s): e2["real"][str(s)] for s in v1_seeds if e2 and str(s) in e2["real"]}
    rows_b = [rows[str(s)] for s in block if str(s) in rows]
    da = [r["frozen"] - r["learn"] for r in rows_b]
    dc = [r["shuf"] - r["learn"] for r in rows_b]
    a["ev2"] = {
        "t": [ttest(da), ttest(dc)],
        "k": [sum(x > 0 for x in da), sum(x > 0 for x in dc)],
        "d": [da, dc],
        "full": len(da) == N_SEEDS,
        "mde": MDE_F * sd(da),
    }
    a["ev2"]["met"] = (
        bool(pc_ok)
        and a["ev2"]["full"]
        and a["ev2"]["t"][0] > BAR
        and a["ev2"]["t"][1] > BAR
        and min(a["ev2"]["k"]) >= K_MIN
    )
    a["ev3"] = any(
        flips(runs[("real", s)]["series"]["action"], runs[("vfrozen", s)]["series"]["action"])[0]
        > floor
        for s in have("real", "vfrozen", among=block)
    )
    nabove, devs = 0, []
    for s in have("real", "pfrozen", among=block):
        nabove += (
            flips(runs[("real", s)]["series"]["action"], runs[("pfrozen", s)]["series"]["action"])[
                0
            ]
            > floor
        )
        devs.append(max(abs(x - 0.1) for x in runs[("real", s)]["series"]["plan_w"]))
    full1 = len(have("real", "pfrozen", among=block)) == N_SEEDS
    a["ep1"] = {
        "n": nabove,
        "of": len(block),
        "full": full1,
        "met": full1 and nabove >= ep1_min,
        "unchanged": bool(devs) and max(devs) <= 0.01,
    }
    p1 = [mr_("real", s) - mr_("pfrozen", s) for s in have("real", "pfrozen", "pshuf", among=block)]
    p2 = [mr_("real", s) - mr_("pshuf", s) for s in have("real", "pfrozen", "pshuf", among=block)]
    a["ep2"] = {
        "t": [ttest(p1), ttest(p2)],
        "k": [sum(x > 0 for x in p1), sum(x > 0 for x in p2)],
        "sd_diff": sd(p1),
        "full": len(p1) == N_SEEDS,
        "d": [p1, p2],
    }
    a["ep2"]["met"] = (
        a["ep2"]["full"]
        and a["ep2"]["t"][0] > BAR
        and a["ep2"]["t"][1] > BAR
        and min(a["ep2"]["k"]) >= K_MIN
    )
    a["ep2"]["mde"] = MDE_F * a["ep2"]["sd_diff"]
    hs = {
        "H-P1": harm([d_p1[s] for s in p1_seeds if s in d_p1]),
        "H-P2": harm([mr_("pshuf", s) - mr_("real", s) for s in have("real", "pshuf")]),
        "H-V3": harm([mae_("real", s) - mae_("vfrozen", s) for s in have("real", "vfrozen")]),
        "H-V1": harm([d_v1[s] for s in v1_seeds if s in d_v1]),
        "H-V2": harm([r["learn"] - r["shuf"] for r in rows.values()]),
    }
    claimed = {"H-P1": DH, "H-P2": DR, "H-V1": DM, "H-V2": DM, "H-V3": DM}
    for k, h in hs.items():
        needs_ctl, n_k = k in ("H-V1", "H-V2"), n_exp.get(k, ns)
        if not valid:
            h["status"] = status(h, claimed[k], ok=False, n_seeds=n_k, why=void_why)
        else:
            h["status"] = status(h, claimed[k], ok=not needs_ctl or ctl_ok, n_seeds=n_k)
        h["met"] = met(h, n_k) and valid and (not needs_ctl or ctl_ok)
        if k in SEQ and valid and h["n"] == n_k and (not needs_ctl or ctl_ok):
            # a verdict is owed (complete, valid, controls PASS): the sequential rule of addenda 2 and 4 decides it,
            # and only once no validity check is pending (re-panel A-M2)
            if pending:
                h["status"] = f"PENDING (validity checks not yet run: {'; '.join(pending[:3])})"
                h["met"] = False
            else:
                h["seq"] = seq_record(h["d"], **SEQ[k])
                h["status"], h["met"] = h["seq"]["status"], h["seq"]["decision"] == "harm"
        elif ns > N_SEEDS and k not in SEQ:
            h["status"], h["met"] = "NO VERDICT (secondary endpoints are decided at look 1)", False
        if h["sd"] == 0:  # A-m4: name the arm whose d_s carry no variance
            h["status"] += f" [{ZERO_SD}: {k}, arm {ARM_OF[k]}]"
    a["hs"] = hs
    # A-M3, R3-M1: every seed of the look; past H-P1's stop a seed runs `real` only
    degenerate = [
        s for s in have("real") if all(mr_(x, s) < SENSITIVITY_R for x in ARMS if (x, s) in runs)
    ]
    a["degenerate"] = degenerate
    a["sens"] = {}
    by_seed = {
        "H-P1": {s: d_p1[s] for s in p1_seeds if s in d_p1},
        "H-V1": {s: d_v1[s] for s in v1_seeds if s in d_v1},
    }
    for k, per_seed in by_seed.items():
        dropped = [s for s in degenerate if s in per_seed]
        h = harm([d for s, d in per_seed.items() if s not in dropped])
        # A-M3: the bar of the decision the flag guards -- the stopped look's t boundary and sign count when a
        # sequential decision exists, the fixed-sample BAR / K_MIN otherwise
        q = hs[k].get("seq")
        bar, k_min = (q["bound_t"], q["k_min"]) if q else (BAR, K_MIN)
        full = hs[k]["t"] > bar and hs[k]["k"] >= k_min
        without = h["t"] > bar and h["k"] >= k_min
        a["sens"][k] = {
            "t": h["t"],
            "k": h["k"],
            "n": h["n"],
            "dropped": dropped,
            "seed_sensitive": bool(dropped) and full != without,
        }
    a["in_brain"] = harm(
        [mr_("real", s) - mr_("vfrozen", s) for s in have("real", "vfrozen", among=block)]
    )
    if a["ev1"]["met"] and a["ev2"]["met"]:
        vd = "CHARGED"
    elif a["ev2"]["met"]:
        vd = "learns its objective in replay; in-brain effect unresolved"
    elif a["ev3"]:
        vd = "charges behaviour, not its objective"
    else:
        vd = "DORMANT"
    if a["ep1"]["met"] and a["ep2"]["met"]:
        pd = "CHARGED"
    elif a["ep1"]["met"]:
        pd = "charges behaviour, objective unresolved (see power note)"
    else:
        pd = "DORMANT"
    if not (a["ev1"]["full"] and a["ev2"]["full"]):
        vd = "NO VERDICT (fewer than the complete seeds)"
    if not (a["ep1"]["full"] and a["ep2"]["full"]):
        pd = "NO VERDICT (fewer than the complete seeds)"
    a["disp"] = {"ValenceLearner": vd, "_plan_w": pd}
    a["wall"] = sum(d["run_s"] for d in runs.values())
    a["n_runs"] = len(runs)
    return a


def render(a):
    """Markdown for one tier (E1's REPORT-TABLES shape plus the E1b harm block). Pure formatting of `analyse`."""
    tier = a["tier"]
    L = [
        f"N={a['N']}  H = ticks {a['H0'] + 1}..{a['N']} (0-based index >= {a['H0']})  bar={BAR}  tier={tier}\n",
        "## Harness validity\n",
    ]
    L += [f"- {v}" for v in a["val"]]
    L.append(f"- **VALIDITY: {'ALL HOLD' if a['valid'] else 'VOID -- ' + '; '.join(a['bad'])}**")
    if a["pending"]:
        L.append(
            f"- PENDING (partial campaign; endpoints with incomplete pairs print NO VERDICT): {'; '.join(a['pending'])}"
        )
    L.append("")
    ev1, ev2, ep2 = a["ev1"], a["ev2"], a["ep2"]

    def word(e):
        return ("MET" if e["met"] else "NOT MET") if e["full"] else "NO VERDICT (incomplete seeds)"

    L.append(
        f"## E-V1\nt(vfrozen-real)={ev1['t'][0]:.3f} ({ev1['k'][0]} seeds >0); t(vshuf-real)={ev1['t'][1]:.3f} ({ev1['k'][1]}) -> **{word(ev1)}**\n"
    )
    L.append(
        f"## E-V2 (replay)\nt(b-a)={ev2['t'][0]:.3f} ({ev2['k'][0]}); t(c-a)={ev2['t'][1]:.3f} ({ev2['k'][1]}); planted control {'PASS' if a['pc_ok'] else 'FAIL/absent'} -> **{word(ev2)}**\n"
    )
    L.append(
        f"## E-V3\nany action flips above the repeat floor ({a['floor']}): {a['ev3']} (reported, no disposition)\n"
    )
    L.append(
        f"## E-P1\nseeds with flips > floor: {a['ep1']['n']}/{a['ep1']['of']} -> **{word(a['ep1'])}**"
        + (
            "; _plan_w never leaves init+/-0.01: uncharged by construction"
            if a["ep1"]["unchanged"]
            else ""
        )
        + "\n"
    )
    L.append(
        f"## E-P2\nt(real-pfrozen)={ep2['t'][0]:.3f} ({ep2['k'][0]}); t(real-pshuf)={ep2['t'][1]:.3f} ({ep2['k'][1]}) -> **{word(ep2)}**; observed mde {ep2['mde']:.4f} vs claimed {DR}\n"
    )
    L.append(
        f"## E1b harm endpoints (tier {tier}; positive d = the neutered/frozen arm is better)\n"
    )
    for k, h in a["hs"].items():
        L.append(
            f"- {k}: t={h['t']:.3f} ({h['k']}/{h['n']} seeds > 0), SD={h['sd']:.4f}; per-seed d = "
            + ", ".join(f"{x:+.4f}" for x in h["d"])
            + f" -> **{h['status']}**"
        )
        if "seq" in h:
            q = h["seq"]
            L.append(
                f"  - sequential decision of record (addendum 2: the first stop{NO_ENC_NOTE if a['tier'] == 'no-encoder' else ''}): "
                f"look {q['look']} of {q['n_looks']}, n={q['n']}: t={q['t']:.3f} vs boundary "
                f"{q['bound_t']:.3f} (z {q['bound_z']:.3f}), {q['k']}/{q['n']} > 0 (need {q['k_min']}), conditional "
                f"power {'n/a' if q['cp'] is None else format(q['cp'], '.3f')} (futility < {FUTILITY_CP}, "
                f"{'open' if q['futility_open'] else 'closed'} at this look) -> {q['decision']}"
            )
            L.append(
                f"  - design power is borrowed: SD {SD_PRE[k]} is E1's no-encoder SD (no production-tier SD exists); "
                f"the observed SD at the stopped look decides powered (re-panel B-N3)"
            )
    n_s = len(a["seeds"])
    L.append(
        "- sign-count companions (no bar), one-sided binomial: "
        + "; ".join(
            f"{k}/{n_s} -> p {sum(math.comb(n_s, j) for j in range(k, n_s + 1)) / 2**n_s:.3f}"
            for k in range(K_MIN, n_s + 1)
        )
    )
    ib = a["in_brain"]
    L.append(
        f"- mde factor: the pre-registered rule {MDE_F} decides; exact noncentral-t cross-check {mde_exact_factor(n_s):.3f}"
    )
    L.append(
        "- E-V1 per-seed (vfrozen-real, vshuf-real): "
        + "; ".join(", ".join(f"{x:+.4f}" for x in d) for d in ev1["d"])
    )
    L.append(
        "- E-V2 per-seed (b-a, c-a): "
        + "; ".join(", ".join(f"{x:+.4f}" for x in d) for d in ev2["d"])
    )
    L.append(
        "- E-P2 per-seed (real-pfrozen, real-pshuf): "
        + "; ".join(", ".join(f"{x:+.4f}" for x in d) for d in ep2["d"])
    )
    L.append(
        "- r_vfrozen - r_real per seed (reported, no bar): "
        + ", ".join(f"{-x:+.4f}" for x in ib["d"])
    )
    L.append(
        f"- in-brain check for ValenceLearner: t(r_real - r_vfrozen) = {ib['t']:.3f} (NOT > {BAR} is required before 'harm-deletion candidate for the learning update')"
    )
    L.append(
        f"- degenerate seeds (mean reward < {SENSITIVITY_R} in every arm it ran): {a['degenerate']}"
    )
    for k, sv in a["sens"].items():
        L.append(
            f"- sensitivity {k} without degenerate seeds {sv['dropped']}: t={sv['t']:.3f} ({sv['k']}/{sv['n']})"
            + ("; **seed-sensitive**" if sv["seed_sensitive"] else "")
        )
    L.append("\n## Dispositions\n")
    L.append(
        f"- ValenceLearner: **{a['disp']['ValenceLearner']}**; `_plan_w`: **{a['disp']['_plan_w']}**"
    )
    L.append(
        f"- E1b `_plan_w` (tier {tier}): H-P1 {a['hs']['H-P1']['status']}; combine tiers ONLY with `cross`"
    )
    L.append(
        f"- E1b ValenceLearner (tier {tier}): H-V1 {a['hs']['H-V1']['status']}; H-V2 {a['hs']['H-V2']['status']}; E-V2 {'MET' if ev2['met'] else 'not met'}"
    )
    L.append(
        "- NULL-RECORD precondition (per method): E-P2/E-V2 not met, mde <= the frozen literal, H-P1/H-V1 not MET, controls PASS, on BOTH tiers, and the production tier's sequential stop powered at its stopped seed count (addendum 3); any VOID or tier-split writes none"
    )
    L.append(f"- perception tier of every number above: {tier} (reported {WANT[tier][0]})")
    L.append(
        "- **If VALIDITY is not ALL HOLD every verdict above is VOID.**"
        if not a["valid"]
        else "- validity ALL HOLD: verdicts stand."
    )
    L.append(f"\nwall: total run_s over runs = {a['wall']:.0f}s ({a['n_runs']} runs)")
    return "\n".join(L)


# ------------------------------------------------------------------------------------ cross-tier
def _series_differ(ra, rb, h0):
    if ra["series"]["action"] != rb["series"]["action"]:
        return True
    ia = [ra["v"]["r"][i] for i in _hmask(ra, h0)]
    ib = [rb["v"]["r"][i] for i in _hmask(rb, h0)]
    return ia != ib


# reuses: seq_record (the first stop over the look prefixes); graph-exists hits (sleep_plasticity_ab, pool_selection,
#   gradient_claim_gate, ...) are term overlap only, none walks sequential looks (sites read)
def _n_owed(key, seeds, d_by_seed):
    """Seeds a primary's verdict reads at this look (A-M4): its own stop of record when a complete look prefix has
    stopped, else this look or its cap, whichever is smaller (addenda 4-9). Runs past a stop are never owed."""
    looks, owed = SEQ[key]["looks"], min(len(seeds), SEQ[key]["looks"][-1])
    done = next((i for i, s in enumerate(seeds) if s not in d_by_seed), len(seeds))
    reach = [n for n in looks if n <= min(done, owed)]
    if reach:
        q = seq_record([d_by_seed[s] for s in seeds[: reach[-1]]], **SEQ[key])
        if q["decision"] != "continue":
            return q["n"]
    return owed


def _agreement(pa, na, key):
    # over the seeds both tiers read: each tier is carried from its own stop (A-M4)
    n = min(pa["n_owed"][key], na["n_owed"][key])
    dp, dn = pa["hs"][key]["d"], na["hs"][key]["d"]
    if len(dp) < n or len(dn) < n:
        return {"n": 0, "of": n, "pearson": float("nan"), "table": []}
    dp, dn = dp[:n], dn[:n]
    return {
        "n": sum((x > 0) == (y > 0) for x, y in zip(dp, dn, strict=True)),
        "of": n,
        "pearson": pearson(dp, dn),
        "table": [
            (sd_, x, y, (x > 0) == (y > 0))
            for sd_, x, y in zip(pa["seeds"][:n], dp, dn, strict=True)
        ],
    }


def cross(pa, pruns, na, nruns):
    """Both tiers, from their `analyse` results and run dicts. Addendum 3 (SINGLE_TIER_HARM_DELETES): the production tier
    decides, agreement and the tier effect are reported beside. Otherwise the evidence file's sections 3, 4 and 6: sign agreement is
    computed per endpoint (H-P1 and H-V1 each use their own d_s), both tiers must share ONE tree sha, and the tier effect
    must be present before any cross-tier wording is used."""
    seeds = pa["seeds"]
    agree = {k: _agreement(pa, na, k) for k in ("H-P1", "H-V1")}
    differ = [
        s
        for s in seeds
        if ("real", s) in pruns
        and ("real", s) in nruns
        and _series_differ(pruns[("real", s)], nruns[("real", s)], pa["H0"])
    ]
    real_complete = all(("real", s) in pruns and ("real", s) in nruns for s in seeds)
    # the look-1 counts scale with the seeds a statement reads: the look, or an endpoint's own cap
    need = lambda m, n=None: math.ceil(m * (len(seeds) if n is None else n) / N_SEEDS)  # noqa: E731
    effect = real_complete and len(differ) >= need(TIER_EFFECT_MIN)
    tree_ok = len(pa["shas"]) == 1 and pa["shas"] == na["shas"]
    out = {
        "agree": agree["H-P1"]["n"],
        "agree_v1": agree["H-V1"]["n"],
        "agree_of": agree["H-P1"]["of"],
        "agree_v1_of": agree["H-V1"]["of"],
        "pearson": agree["H-P1"]["pearson"],
        "pearson_v1": agree["H-V1"]["pearson"],
        "tier_effect": effect,
        "differ_seeds": differ,
        "tree_ok": tree_ok,
        "table": agree["H-P1"]["table"],
        "table_v1": agree["H-V1"]["table"],
        "need_agree": need(AGREE_MIN, agree["H-P1"]["of"]),
        "need_agree_v1": need(AGREE_MIN, agree["H-V1"]["of"]),
        "need_effect": need(TIER_EFFECT_MIN),
    }

    def v1_tail(tiers):
        """The in-brain check (frozen section 6) on each COUNTED tier: complete pairs and t(r_real - r_vfrozen) not > BAR."""
        ib = [a["in_brain"] for a in tiers]
        if any(
            x["n"] != N_SEEDS for x in ib
        ):  # vfrozen runs exist for the look-1 block only (addendum 2)
            return ": in-brain check incomplete (no full vfrozen pairs): objective finding only, owner decides"
        if all(not (x["t"] > BAR) for x in ib):
            return ": in-brain check holds, harm-deletion candidate for the learning update; owner decides"
        return (
            ": objective finding (replay harm, in-brain consequence present/unclear), owner decides"
        )

    def single_tier(key, tail):
        prod, other = pa["hs"][key]["status"], na["hs"][key]["status"]
        if pa["hs"][key]["met"]:
            beside = f"no-encoder {other}, d_s sign agrees on {agree[key]['n']}/{agree[key]['of']}"
            beside += "" if effect else ", tiers indistinguishable on this path"
            return f"harm on the production tier ({prod}; the confirmatory tier; {beside})" + tail(
                (pa,)
            )
        if na["hs"][key]["met"]:
            return f"tier finding: harm on no-encoder only; the production tier decides ({prod}): no deletion candidate (owner decides)"
        if (
            pa["hs"][key].get("seq", {}).get("decision") == "continue"
        ):  # never "not met" (addendum 2)
            return f"CONTINUE on the production tier (no verdict): {prod}; beside it no-encoder {other}"
        return f"not met on the production tier: {prod}; beside it no-encoder {other}"

    def verdict(key, tail):
        """`tail(tiers)` words what the counted tiers allow; both tiers for 'harm on both', production alone for the flag."""
        ma, mb = pa["hs"][key]["met"], na["hs"][key]["met"]
        n_agree = agree[key]["n"]
        if SINGLE_TIER_HARM_DELETES and not pa["valid"]:  # a verdict line, not a `refuse` exit
            return "REFUSED (the production tier has no verdict: it alone decides, a no-encoder result never deletes)"
        # #4616 "production tier decides" (C-M3, A-m6; re-panel B-N4): under addendum 3 the no-encoder tier is reported
        # beside in EVERY state -- VOID, short, pending, or on another tree -- and never holds or vetoes the decision
        beside = SINGLE_TIER_HARM_DELETES
        counted = (("production", pa),) if beside else (("production", pa), ("no-encoder", na))
        if not all(a["valid"] for _, a in counted):
            who = [t for t, a in counted if not a["valid"]]
            return f"PENDING (no verdict on tier {', '.join(who)}; a tier with no verdict makes every cross-tier statement PENDING)"
        if not (tree_ok or beside):
            return f"PENDING (the two tiers do not share one recorded tree sha: {pa['shas']} vs {na['shas']})"
        short = [t for t, a in counted if a["hs"][key]["n"] != a["n_owed"][key]]
        real_ok = all(("real", s) in pruns for s in seeds) if beside else real_complete
        if short or not real_ok:
            return f"PENDING (endpoint {key} has no verdict on tier {', '.join(short) or 'either (real series incomplete)'})"
        if not effect and not SINGLE_TIER_HARM_DELETES:
            return "tiers indistinguishable on this path: no cross-tier claim, every tier result is one replicate"
        held = [p for _, a in counted for p in a["pending"]]
        if SINGLE_TIER_HARM_DELETES and pa["pending"]:
            return (
                f"PENDING (validity checks not yet run on the production tier: {'; '.join(pa['pending'][:3])}): "
                "no verdict and no candidate wording yet"
            )
        if held and (ma or mb):
            return (
                f"{key} MET on {'both tiers' if ma and mb else 'production' if ma else 'no-encoder'} but PENDING "
                f"(validity checks not yet run: {'; '.join(held[:3])}): no tier finding and no candidate wording yet"
            )
        if SINGLE_TIER_HARM_DELETES:
            ne = [
                "VOID: reported beside, no verdict" if not na["valid"] else "",
                "" if tree_ok else f"on another tree ({na['shas']}): reported beside",
                f"short ({na['hs'][key]['n']}/{na['n_owed'][key]} seeds): reported beside"
                if na["valid"] and na["hs"][key]["n"] != na["n_owed"][key]
                else "",
                f"checks pending ({'; '.join(na['pending'][:3])}): reported beside"
                if na["valid"] and na["pending"]
                else "",
            ]
            line = single_tier(key, tail)
            return "; ".join([line] + [f"no-encoder {x}" for x in ne if x])
        if ma and mb:
            if n_agree >= need(AGREE_MIN, agree[key]["of"]):
                return (
                    "harm on both tiers (effective p between .0025 and .05, correlated tiers)"
                    + tail((pa, na))
                )
            return f"both tiers MET but d_s sign agrees on only {n_agree}/{agree[key]['of']}: tier finding, no deletion candidate"
        if ma or mb:
            return f"tier finding: harm on {'production' if ma else 'no-encoder'} only; no deletion candidate (owner decides)"
        return f"not met on either tier: production {pa['hs'][key]['status']}; no-encoder {na['hs'][key]['status']}"

    out["h_p1"] = verdict(
        "H-P1",
        lambda tiers: (
            ": the learned `_plan_w` trajectory is a harm-deletion candidate; owner decides"
        ),
    )
    out["h_v1"] = verdict("H-V1", v1_tail)
    out["null_pre"] = {m: null_precondition(m, pa, na) for m in ("_plan_w", "ValenceLearner")}
    return out


def null_precondition(method, pa, na):
    """The ONE NULL-RECORD per method (addendum 3): a measured harm on either tier writes none (production: the harm
    route; no-encoder: a tier finding); then BOTH tiers need a verdict, controls PASS and the objective endpoint null at
    adequate power; and the production primary needs a stop of record, powered at its stopped n. Returns (ok, why); the
    record itself is the integrator's (needs the arc sha)."""
    if method == "_plan_w":
        key, lit, obj = "H-P1", DR, lambda a: (not a["ep2"]["met"], a["ep2"]["mde"])
    else:
        key, lit, obj = "H-V1", DM, lambda a: (not a["ev2"]["met"], a["ev2"]["mde"])
    if pa["hs"][key]["met"]:
        return (
            False,
            f"production: {key} MET (a measured harm gets the harm route, never a null record)",
        )
    if na["hs"][key]["met"]:
        return False, f"no-encoder: {key} MET (a tier finding, no null record)"
    if not (pa["valid"] and na["valid"]) or pa["pending"] or na["pending"]:
        return False, "a tier has no verdict (VOID/PENDING)"
    if len(pa["shas"]) != 1 or pa["shas"] != na["shas"]:
        return False, "the two tiers do not share one recorded tree sha"
    for t, a in (("production", pa), ("no-encoder", na)):
        not_met, mde = obj(a)
        if not a["ctl_ok"]:
            return False, f"{t}: a planted control failed"
        if not not_met:
            return False, f"{t}: the objective endpoint was met"
        if not mde <= lit:
            return False, f"{t}: observed mde {mde:.4f} > frozen literal {lit} (INCONCLUSIVE)"
    # the production stop of record decides; a no-encoder CONTINUE is reported only
    q = pa["hs"][key].get("seq")
    if q is None:
        return False, f"production: {key} has no sequential stop (no verdict): no null"
    if q["decision"] == "continue":
        return (
            False,
            f"production: {key} sequential design still running (CONTINUE): no null before a stop",
        )
    if not q["powered"]:
        return (
            False,
            f"production: {key} stopped at n={q['n']}, mde {q['mde']:.4f} above the claimed effect: INCONCLUSIVE",
        )
    return (
        True,
        f"production stopped powered at n={q['n']}; both tiers null on the objective, no harm, controls PASS",
    )


def bridge_readout(bridge_dir, e1_dir=None, seeds=PSEEDS, tree_shas=None):
    """BRIDGE: new tree, E1's seeds, no-encoder tier vs E1's own recorded real/pfrozen series (read-only). The bridge
    runs carry the same validity checks as a campaign run and must share the campaign's one tree sha."""
    e1_dir = e1_dir or E1_RESULTS
    new = {
        (a, s): load_run(bridge_dir, "no-encoder", f"{a}_{s}")
        for s in seeds
        for a in ("real", "pfrozen")
    }
    present = {k: d for k, d in new.items() if d}
    rows, dr, same_all = [], [], True
    for (arm, s), d in new.items():
        p = os.path.join(e1_dir, f"{arm}_{s}.json")
        if d is None or not os.path.exists(p):
            rows.append((arm, s, None))
            continue
        with open(p, encoding="utf-8") as f:
            old = json.load(f)
        same = d["v"]["r"] == old["v"]["r"] and d["series"]["action"] == old["series"]["action"]
        same_all &= same
        rows.append((arm, s, same))
    for s in seeds:
        if ("real", s) in present and ("pfrozen", s) in present:
            h0 = present[("real", s)]["ticks"] // 2
            dr.append(mean_r(present[("pfrozen", s)], h0) - mean_r(present[("real", s)], h0))
    problems = []
    for (arm, s), d in present.items():
        problems += run_problems(arm, s, d, "no-encoder")
        if not d["tier"]["tree_clean"]:
            problems.append(f"{arm}_{s}: tree_clean = false")
    shas = sorted({d["tier"]["tree_sha"] for d in present.values()})
    if tree_shas is not None and (len(shas) != 1 or shas != list(tree_shas)):
        problems.append(f"bridge tree sha {shas} is not the campaign's {list(tree_shas)}")
    return {
        "complete": all(r[2] is not None for r in rows),
        "tree_equivalent": same_all,
        "tree_ok": not problems,
        "problems": problems,
        "shas": shas,
        "rows": rows,
        "bridge_h_p1": harm(dr),
        "e1_t": E1_HP1_T,
    }


def render_cross(c, br=None):
    gates = (
        ("reported beside, no gate (addendum 3)",) * 3
        if SINGLE_TIER_HARM_DELETES
        else (
            f"required >= {c['need_agree']}",
            f"required >= {c['need_effect']}",
            f"required >= {c['need_agree_v1']}",
        )
    )
    L = [
        "## Cross-tier (e1b-rpe-harm.md sections 3, 4, 6; addendum 3)\n",
        "| seed | d_s production | d_s no-encoder | sign agrees |",
        "|---|---|---|---|",
    ]
    L += [f"| {s} | {x:+.4f} | {y:+.4f} | {ok} |" for s, x, y, ok in c["table"]]
    L.append(
        f"\nsign agreement {c['agree']}/{c['agree_of']} ({gates[0]}); Pearson r of d_s across tiers {c['pearson']:.3f} (reported, no bar)"
    )
    L.append(
        f"tier effect present: {c['tier_effect']} (same-seed `real` series differ on {len(c['differ_seeds'])} seeds; {gates[1]})"
    )
    L.append(
        f"H-V1 own d_s sign agreement {c['agree_v1']}/{c['agree_v1_of']} ({gates[2]}); Pearson r {c['pearson_v1']:.3f}"
    )
    L.append(f"- `_plan_w` H-P1: **{c['h_p1']}**\n- ValenceLearner H-V1: **{c['h_v1']}**")
    L += [f"- null_precondition {m}: {ok} ({why})" for m, (ok, why) in c["null_pre"].items()]
    if br and not br["complete"]:
        L.append("\nBRIDGE incomplete: some runs are missing; no statement about E1's signal")
    elif br and not br["tree_ok"]:
        L.append(f"\nBRIDGE VOID: {'; '.join(br['problems'][:3])}")
    elif br:
        h = br["bridge_h_p1"]
        L.append(
            f"\nBRIDGE (E1 seeds, new tree, no-encoder): {'tree-equivalent' if br['tree_equivalent'] else 'tree moved the run'}; H-P1 t={h['t']:.3f} ({h['k']}/{h['n']}) beside E1's t +{br['e1_t']}"
        )
    else:
        L.append(
            "\nBRIDGE not run: tree and seeds both differ from E1, no statement about E1's signal"
        )
    return "\n".join(L)


# ---------------------------------------------------------------------------- brain-free fixtures
def synth_run(
    arm,
    seed,
    tier="production",
    *,
    r=0.0,
    mae_value=0.0,
    ticks=N_TICKS,
    wiggle=0.0,
    shift=0,
    sha=SYNTH_SHA,
):
    """A constructed run with only the fields compare reads. Constant reward r (plus a zero-mean +-wiggle) and a
    constant |pred - r| = mae_value over every tick; `shift` moves the action series so arms can flip against `real`."""
    n = ticks
    rs = [r + (wiggle if i % 2 else -wiggle) for i in range(n)]
    pred = [x - mae_value for x in rs]
    t = {
        "requested": tier,
        "reported": WANT[tier][0],
        "perceiving_real": WANT[tier][1],
        "perceiving_audio": WANT[tier][2],
        "state_api": WANT[tier][0],
        "pins": {},
        "sha": dict(ENC) if tier == "production" else {},
        "pc_cortex_tier": "stdlib",
        "optimize": 0,
        "tree_sha": sha,
        "tree_clean": True,
        "encoder_calls": {"vision": n, "audio": n, "pc": 0}
        if tier == "production"
        else {"vision": 0, "audio": 0, "pc": 0},
        "percept_ticks": {"vision": n, "audio": n}
        if tier == "production"
        else {"vision": 0, "audio": 0},
    }
    return {
        "arm": arm,
        "seed": seed,
        "ticks": n,
        "errs": 0,
        "run_s": 1.0,
        "swallowed_exceptions": 0,
        "exceptions": [],
        "learn_calls": n,
        "foreign_valence": 0,
        "foreign_learn": 0,
        "pfc_bias_calls": n,
        "pc_stub_calls": n if arm == "pc" else 0,
        "pattern_len_ok": True,
        "shuffle_wrapped": False,
        "perm_len": None,
        "plan_lr_final": 0.0 if arm == "pfrozen" else 0.02,
        "persist": False,
        "nord_persist_env": None,
        "tier": t,
        "v": {
            "tick": list(range(n)),
            "r": rs,
            "pred": pred,
            "rpe": [mae_value] * n,
            "pattern": [[0.0]] * n,
        },
        "p": {"rpe_entry": [0.0] * n, "rpe_used": [0.0] * n},
        "series": {
            "action": [(i + seed + shift) % 3 for i in range(n)],
            "plan_w": [0.1 + 0.001 * (i % 7) for i in range(n)],
        },
    }


def fixture_runs(kind, tier="production"):
    """kind: harm (pfrozen = real + 0.1 with seed-varying noise) | noharm (pfrozen = real) | void (same data as harm)."""
    wig = 0.01 if tier == "production" else 0.0
    runs = {}
    for s in SEEDS:
        base = 0.2 + 0.01 * (s % 5)
        # spread must stay small: the planted harm must cross H-P1's look-1 boundary
        noise = 0.001 * ((s * 7) % 5 - 2)
        dh = 0.1 + noise if kind in ("harm", "void") else 0.0
        for arm in ARMS:
            r = base + (dh if arm in ("pfrozen", "pshuf") else 0.0)
            runs[(arm, s)] = synth_run(
                arm,
                s,
                tier,
                r=r,
                mae_value=0.4 + 0.02 * (s % 3),
                wiggle=wig,
                shift=1 if arm != "real" else 0,
            )
    s0 = SEEDS[0]
    runs[("real_r2", s0)] = copy.deepcopy(runs[("real", s0)])
    for s in (s0, s0 + 1):
        runs[("pc", s)] = synth_run("pc", s, tier, r=0.2, mae_value=0.4, wiggle=wig, shift=2)
    return runs


def fixture_e2(kind="ok", tier="production"):
    """A brain-free replay result: synthetic per-seed rows plus the planted controls ('void' fails the learning control).
    Each row is bound to the `fixture_runs` real run of its seed on `tier` (the same series for every kind)."""
    real = fixture_runs("noharm", tier)
    rows = {
        str(s): {
            "learn": 0.5 + 0.01 * (s % 4),
            "frozen": 0.5 + 0.01 * (s % 4) - 0.02 * ((s * 3) % 5 - 1),
            "shuf": 0.55,
            "n_calls": N_TICKS,
            "N": N_TICKS,
            "n_scored": 150,
            **replay_binding(real[("real", s)]),
        }
        for s in SEEDS
    }
    return {
        "real": rows,
        "planted": planted_rows("null" if kind == "void" else "learn"),
        "planted_harm": planted_rows("harm"),
    }


def fixture_lines():
    """The three fixture verdicts the code prints now (expected: `FIXTURE_WANT`)."""
    out = []
    for kind in ("harm", "noharm", "void"):
        a = analyse(
            "production",
            fixture_runs(kind),
            fixture_e2("void" if kind == "void" else "ok"),
            logs_pass=True,
        )
        out.append(
            f"fixture {kind} H-P1 {a['hs']['H-P1']['status']} | H-V1 {a['hs']['H-V1']['status']}"
        )
    return out


# --------------------------------------------------------------------------------------- cli / self-check
def _analyse_dir(d, tier, seeds=SEEDS):
    """results/<tier> -> (analysis, runs); the replay record and the logs gate live beside/above it."""
    e2p = os.path.join(d, "e2v2.json")
    e2 = None
    if os.path.exists(e2p):
        with open(e2p, encoding="utf-8") as f:
            e2 = json.load(f)
        if (e2.get("tier") or {}).get("requested") not in (None, tier):
            raise SystemExit(f"TIER MISMATCH in {e2p}: {e2['tier']}")
    runs = load_runs(d, tier, seeds)
    lg = logs_gate(os.path.join(d, "..", "..", "logs"))
    return analyse(tier, runs, e2, seeds=seeds, logs_pass=lg), runs


def _emit(txt, dest):
    print(txt)
    if dest:
        with open(dest[0], "w", encoding="utf-8") as f:
            f.write(txt)


def main(argv=()):
    argv = list(argv)
    if not argv:
        return _selfcheck()
    cmd, rest = argv[0], argv[1:]
    seeds = SEEDS
    if "--look" in rest:  # addenda 2, 4, 5, 7, 9
        i = rest.index("--look")
        look = int(rest[i + 1])
        if not 1 <= look <= len(CAMPAIGN_LOOKS):
            refuse("any", cmd, f"--look {look} is not one of 1..{len(CAMPAIGN_LOOKS)}")
        seeds, rest = (
            range(SEEDS[0], SEEDS[0] + CAMPAIGN_LOOKS[look - 1]),
            rest[:i] + rest[i + 2 :],
        )
    if cmd == "run":
        import argparse

        ap = argparse.ArgumentParser(prog="rpe_charge run")
        ap.add_argument("--arm", required=True, choices=[*ARMS, "pc"])
        ap.add_argument("--seed", type=int, required=True)
        ap.add_argument("--ticks", type=int, default=N_TICKS)
        ap.add_argument("--tier", required=True, choices=sorted(WANT))
        ap.add_argument("--home", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--series", default="")
        a = ap.parse_args(rest)
        run_arm(a.arm, a.seed, a.ticks, a.tier, a.home, a.out, series=a.series or None)
        sys.stdout.flush()
        os._exit(0)  # measurement only: skip atexit persistence
    if cmd == "replay":
        d, tier = rest[0], rest[1]
        n = int(rest[2]) if len(rest) > 2 else None
        runs = {s: load_run(d, tier, f"real_{s}") for s in seeds}
        out = replay({s: r for s, r in runs.items() if r}, n)
        out["tier"] = next((r["tier"] for r in runs.values() if r), None)
        with open(os.path.join(d, "e2v2.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print(json.dumps(out["real"], indent=1))
        return 0
    if cmd == "compare":
        a, _ = _analyse_dir(rest[0], rest[1], seeds)
        _emit(render(a), rest[2:])
        return 0
    if cmd == "cross":
        (pa, pruns), (na, nruns) = (
            _analyse_dir(rest[0], "production", seeds),
            _analyse_dir(rest[1], "no-encoder", seeds),
        )
        bd = os.path.join(os.path.dirname(rest[0].rstrip("/\\")), "bridge")
        br = bridge_readout(bd, tree_shas=pa["shas"]) if os.path.isdir(bd) else None
        _emit(render_cross(cross(pa, pruns, na, nruns), br), rest[2:])
        return 0
    if cmd == "fixture":
        os.makedirs(rest[0], exist_ok=True)
        lines = fixture_lines()
        if not fixture_ok(lines):
            print(f"fixture verdicts are not the expected ones, nothing written: {lines}")
            return 1
        with open(os.path.join(rest[0], "fixture.log"), "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        return 0
    print(f"unknown command {cmd!r}")
    return 2


def _selfcheck():
    """Brain-free: planted controls, VOID, the fixtures, the exact-mde cross-check, and the patch scope on a stand-in."""
    import types

    ta, tb, ok = control_pass(planted_rows("learn"))
    if not ok:
        print(f"rpe_charge self-check FAILED: planted-learning control t={ta:.2f}/{tb:.2f}")
        return 1
    if control_pass(planted_rows("null"))[2]:
        print("rpe_charge self-check FAILED: a planted null passed the control")
        return 1
    if not harm_control_pass(planted_rows("harm"))[1] or harm_control_pass(planted_rows("null"))[1]:
        print("rpe_charge self-check FAILED: planted-harm control does not separate harm from null")
        return 1
    h = harm([0.1 + 0.01 * i for i in range(N_SEEDS)])
    if status(h, DH) != "MET" or status(h, DH, ok=False) != "VOID (control failed)":
        print("rpe_charge self-check FAILED: MET / VOID status")
        return 1
    fx = fixture_lines()
    if (
        "H-P1 MET" not in fx[0]
        or "H-P1 CONTINUE at look 1" not in fx[1]
        or "H-V1 VOID" not in fx[2]
    ):
        print(f"rpe_charge self-check FAILED: fixtures {fx}")
        return 1
    f = mde_exact_factor(N_SEEDS, BAR)
    if not (0.9 * MDE_F < f <= MDE_F):
        print(f"rpe_charge self-check FAILED: exact mde factor {f:.4f}")
        return 1
    from snn_data_lake.plasticity.valence_stdp import ValenceLearner, _pattern

    mine, other, orig, win = ValenceLearner(), ValenceLearner(), ValenceLearner.learn, _pattern(1)
    with ArmPatch("vfrozen") as ap:
        ap.bind(types.SimpleNamespace(_valence=mine))
        mine.learn(win, 1.0)
        other.learn(win, 1.0)
        scoped = (
            ap.v["calls"] == 1 and ap.v["foreign_learn"] == 1 and not any(mine.w) and any(other.w)
        )
    if not scoped or ValenceLearner.learn is not orig:
        print("rpe_charge self-check FAILED: vfrozen patch scope / restore")
        return 1
    print(
        f"rpe_charge self-check OK: planted-learning control t={ta:.1f}/{tb:.1f}, planted null refused, planted-harm "
        f"separated, MET/VOID status, 3 fixtures, exact mde factor {f:.3f} (rule {MDE_F}), neuter scoped to the arm "
        "instance and restored. No brain was built; the campaign runs on the pushed head (integrator)."
    )
    return 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main(sys.argv[1:]))
