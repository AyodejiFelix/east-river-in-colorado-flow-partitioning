# East River Tracer-Informed PINN — Consolidated Findings

This document tells the complete diagnostic story of this project, in
order, for a reader who has not opened `pinn/README.md` (the detailed,
chronological, commit-by-commit research log) or any commit message. It
is the research narrative — what was found, what was tried, what worked,
what didn't, and why. For exact commands, code, and per-round figure
sets, see `pinn/README.md` and the `results_*/`, `figures_*/` directories
it references.

**The model**: a physics-informed neural network for the East River
watershed (Colorado), implementing two linear reservoirs per sub-basin
(fast `S_f`, slow `S_s`) driven by a learned partitioning of effective
water input, with an exact closed-form exponential update (not a
numerical ODE stepper) so the water/tracer mass balance is satisfied
exactly regardless of training. Five modeled nodes (Rustlers, Copper,
EAQ, Gothic_ME, PH_LT — the watershed outlet) are combined via a routed
network topology, not a flat sum. Two variants are maintained throughout:
`single_C_in` (one chloride end-member) and `split_C_in` (separate
fast-pathway and groundwater end-members). Trained on CLM forcing,
observed discharge, and observed stream/groundwater chloride, all
included in this repo.

---

## 1. The original three symptoms, and how they were connected

The first fully-trained checkpoints (`results/`, `results_cinsplit/`)
showed three problems that looked, at first, like they might be three
separate bugs:

1. **Chloride concentration spikes** — simulated `C_tot` at the outlet
   spiking to 40-60 mg/L against an observed baseline under 1 mg/L.
2. **`S_s` (slow reservoir) apparently collapsed** — a review hypothesis
   that the slow reservoir was chronically starved near its numerical
   floor, with no physical basis, causing the spikes above.
3. **Unstable fast-fraction** — the partition network's daily
   `f_uf,j(t)`/`f_f,j(t)` outputs flickered erratically day to day, with
   nothing in the loss penalizing how much they changed.

A dedicated diagnostic pass (the `figures/08`–`11` set) traced all three
to **one shared root cause**: the partition network re-decides how to
split water every single day from that day's forcing alone, with no
smoothness constraint. When it swings toward the fast pathway, `S_s`
starves and decays to its floor within days (`K_s` gives a 1-2 week
e-folding time); `Q_tot` then crashes toward the routing floor used in
`C_tot = Phi_tot / Q_tot`, and dividing a small-but-nonzero mass flux by
a near-zero flow produces the concentration spikes. When the network
swings back, `S_s` refills and the whole cycle repeats. **`S_s` was not
"chronically" low — it was bimodal** (36-73% of days within 10x of its
floor, essentially two clusters: near-empty or well-stocked, almost
nothing in between) — and the routed baseflow fraction was not
uniformly *too low* either; it whipsawed between 0 and 100% and its
winter mean was often *too high* (75-88% at 4 of 5 gauges, versus a
12-33%/17-50% literature benchmark — see Section 3). **The problem was
volatility, not a directional bias.** The ODE integrator itself was
independently confirmed exact (matches a 20,000-substep numerical
reference to ~1e-4mm) — the flicker was a property of what was being fed
into a correct integrator, not a numerical-integration bug.

The first candidate fix — a temporal-smoothness penalty on
`f_uf,j(t)`/`f_f,j(t)` — **did not work**, reported plainly rather than
oversold: it cut the raw flicker metric substantially (-58% to -63%) but
left floor-crash rate, chloride spike magnitude, and baseflow fraction
essentially unchanged. The storage-floor diagnostic showed *why*:
`Q_tot` was hitting the floor during **sustained multi-week episodes**,
not single-day noise — a day-to-day smoothness term has nothing to act
on during a multi-week dwell at a low, stable level. This motivated the
storage barrier (Section 2).

---

## 2. The storage barrier: what it fixed, what it couldn't, and why it's not further tunable

**The fix**: a soft penalty, `mean(relu(threshold - S_s)^2)`, discouraging
`S_s` from dwelling near empty — a *level* penalty, not a rate penalty,
targeting the actual failure mode (sustained low storage) directly.
Paired with a separate, more conservative floor specifically on
`C_tot`'s denominator, and a hard ceiling on `C_tot` itself (empirically
derived: the maximum stream chloride ever observed in this repo, ×5
safety margin).

**What it fixed, cleanly and completely**: floor-crash rate collapsed
from 2.8-19.5% at every gauge to 0.0-0.9%. Chloride spikes at the outlet
dropped from a ~60 mg/L peak to a handful of residual days ~1.4-1.7 mg/L
(a real, precisely-characterized ~35x reduction, not full elimination).
Point-fit NSE improved substantially
at most gauges as a side effect (e.g. PH_LT `Cl_NSE`: -220.85 → -0.90).

**What it revealed, not fixed**: baseflow fraction got **worse**, not
better. Winter-mean `Q_s,tot/Q_tot` rose to 89-96% at every gauge — the
barrier mechanism working exactly as specified (keep `S_s` elevated) with
a side effect the specification didn't anticipate: "elevated" turned out
to mean "elevated almost all the time," not just "not pathologically
near-zero."

**Why it's not a tunable-away problem** — three separate sweep rounds,
run at three different points in the project as later fixes changed the
water budget underneath it, all point the same direction:

- **Weight sweep** ({0.1x, 0.3x, 1x, 3x} of the working value):
  floor-crash rate and baseflow fraction essentially flat across the
  entire 30x range. `S_s`'s converged level is weight-invariant, but sits
  40-80x the threshold itself, not near it — the barrier's mere presence
  pushes the system into a much higher regime regardless of how strongly
  it's weighted.
- **Threshold sweep** ({0.1, 0.3, 1.0, 3.0} mm): floor-crash and baseflow
  fraction again flat. But at PH_LT specifically, `S_s`'s converged level
  scales clearly with threshold (15.4mm → 41.4mm, a 2.7x range) with
  **zero** effect on baseflow fraction over that same range — a direct,
  decisive proof that `S_s`'s absolute level is not what sets baseflow
  fraction at all.
- **Re-swept after the phase-partitioning fix** (Section 4) changed the
  water budget: same grid, on the new budget. **No single setting
  satisfies floor-crashing, baseflow fraction, and no chloride
  regression simultaneously** — a direct, consistent trade-off across
  the whole grid (lower threshold → better baseflow fraction, worse
  chloride; higher threshold → better chloride, worse baseflow
  fraction). A follow-up check found **all 5 gauges agree on the same
  preferred setting for both metrics** — there is no gauge-specific
  divergence, so **per-gauge barrier calibration was considered and
  ruled out**, not just left untried (see Section 8).
- **EAQ's floor-crash rate never reached zero at any setting tested**
  (3.1-5.2% throughout). Direct investigation found it genuinely
  connected to the same storage-depletion mechanism (`S_f` empty on
  27/29 floor-crash days; `S_s` below the barrier threshold on 20.7% of
  floor-crash days vs. 0.0% of all other days) — but not fixable by
  re-tuning, because the barrier is a **soft training-time loss
  penalty**, not a hard floor on the state (`S_s`'s actual numerical
  floor is `clamp_min(1e-6)`, effectively unconstrained). It cannot
  inject water that genuinely isn't in the forcing during a real
  multi-day input drought.

**Bottom line**: the barrier is the correct, working, thoroughly-tested
fix for its two target problems (floor-crashing, unbounded
concentration). It is not, and by strong direct evidence cannot be made
to be, a lever on baseflow fraction. That required a different
investigation entirely (Section 3).

---

## 3. The baseflow-fraction investigation: from a recession-rate mismatch to a structural loss-surface finding

With the barrier ruling out `S_s`'s level as the driver, the search
moved to what *does* set baseflow fraction. Two candidates were tested
directly rather than assumed:

**`K_s` (slow-reservoir recession rate)**: checked for saturation
(sigmoid sitting mid-range, 0.068-0.108 — not pinned) and for
gradient-starvation. The gradient check was initially ambiguous because
`train.py` uses a cosine-annealed learning-rate schedule that reaches
exactly zero at the final epoch — a near-zero gradient there is
consistent with either "converged" or "schedule artifact." Resolved
directly: 300 more epochs at a fresh, non-decayed learning rate showed
loss still dropping substantially (-9.5%) and `K_s` moving ~23% at every
gauge — genuinely trainable, not stuck. **But baseflow fraction barely
moved** (-1.3 to +0.4 points) despite that real, substantial parameter
change. Conclusion: not a saturation problem — genuine structural
insensitivity. `K_s` is not the lever.

**Recharge fraction (`1-f_f,j`, how much infiltration the partition
network routes to the slow reservoir)**: tested the same way — forced,
not trained, at fixed values {0.1, 0.3, 0.5, 0.7, 0.9} on the existing
checkpoint, no retraining. **Confirmed as a real lever**: baseflow
fraction moves dramatically and almost linearly with it at every gauge.
`recharge_fraction=0.3` lands winter baseflow fraction inside the
12-33%/17-50% benchmark band at every single gauge. The network's own
freely-learned recharge fraction sits near 0.65-0.9 — squarely in the
part of this curve that produces the observed 84-96% baseflow fraction.

This raised the natural next question: would a bias/initialization nudge
toward `rf≈0.3` work, or would training pull it back? Tested directly,
no new training: computed the **data loss alone** (discharge + chloride,
excluding regularizers) at each forced recharge fraction. **The data
loss itself, not just the model's current position, monotonically and
strongly prefers high recharge fraction** — forced `rf=0.3` costs 140.9%
more data loss than the network's freely-learned setting. This is not a
local-optimum or initialization artifact; the loss surface has a real,
strong slope toward high recharge fraction across the whole tested
range. An init nudge would fight this gradient at every step.

**Why, traced to its root cause**: a master-recession-curve analysis
against the *full* observed discharge record (not just the training
window) found that real winter recessions decay far slower than the
model's fast reservoir constant `K_f` — the observed recession rate sits
10.7x to 54.9x closer to `K_s` than to `K_f` at every gauge, and even
*slower* than `K_s` itself everywhere. **A single fixed-per-basin `K_f`
structurally cannot both drain fast enough for storm response and hold
water through a multi-month winter recession.** Given that conflict,
routing most water away from the fast reservoir (i.e. a high recharge
fraction) is the model's only available way to fit the abundant winter
recession data well — which is exactly what the loss surface measurement
above found directly.

**The fix that followed** — `K_f,j(t)` as a bounded, covariate-dependent
function of forcing intensity (not a free function of time, per the
formulation doc's own caution) — worked exactly as designed at the
mechanism level (verified genuinely time-varying post-training, total
loss roughly halved) but **did not move baseflow fraction** (still
84-96%). The reason, diagnosed rather than assumed: the loss function
(plain squared-error, no peak weighting) is dominated by the far more
numerous low-flow days, so a `K_f(t)` that can now fit those days better
lowers the aggregate loss substantially without needing to reproduce the
rare peak days at all. This pointed toward two further questions —
whether `K_f,storm`'s bound itself could physically produce a realistic
peak, and whether the loss under-weights peak days — both resolved later
(Section 6) and both, ultimately, **not** the answer.

**Where baseflow fraction stands as of the current checkpoint**: it
*did* move substantially once the missing snowmelt term was added
(Section 4) — the single largest move of any fix tried, ~17-24
percentage points at every gauge — but subsequent fixes (phase
partitioning, melt-timing lag) each pulled it back somewhat in the other
direction as a side effect of resolving other bugs. **It remains above
the 12-33% annual benchmark band at every gauge in the current
checkpoint** (72-83% annual — see Section 9). This is the one target
metric from the original investigation that has moved the most but is
still not resolved.

---

## 4. The phase-partitioning fix: the missing-snowmelt bug, the fix, and its tension with the storage barrier

**Finding the bug**: testing whether `K_f,storm`'s rate bound could ever
produce a realistic multi-day snowmelt peak (isolated best-case
reservoir simulation, no partitioning) found that magnitude wasn't
fundamentally capped, but the **input signal itself** was sparse and
intermittent during the actual observed peak windows — only 26-53 of 76
days in each window had any measurable water input at all. Traced to its
cause: `P_e,j = P_j` (CLM precipitation only) was a documented
simplification, diverging from the formulation doc's actual governing
equation `P_e,j = P_j + SM_j` (precipitation **plus snowmelt**), adopted
originally because no SWE data covering the training window existed in
this repo. **Checked the SWE file's coverage before writing any code** —
the first available pull covered 2023-2026 only,
zero overlap with the Dec 2013-Dec 2015 training window; stopped there,
no code written, until an extended pull covering 2013-2015 was supplied
and its coverage independently verified (737/737 training-window days
present).

**The snowmelt term**: `SM_j(t)` computed as day-to-day SWE decline from
the single Butte SNOTEL 380 station (positive decrease = melt, clipped
at zero on accumulation days), applied uniformly to all 5 sub-basins (an
explicit, documented single-station representativeness caveat).
Implementing it moved baseflow fraction the most of any fix in the
project (Section 3) and removed the structural cap on peak magnitude —
but also revealed a second bug: `P_e,j = P_j + SM_j` **double-counts**
water on days precipitation falls as snow, since raw CLM precipitation
doesn't distinguish rain from snow — that day's water was being counted
as immediate liquid input *and*, later, as melt via `SM_j`.

**The fix — rain/snow phase partitioning**: no temperature forcing
exists anywhere in this repo (checked every column of every file), so
the Butte 380 SWE record itself was used as the rain/snow proxy: on any
day SWE increases (snow accumulating, confirmed directly in the record),
that day's precipitation is withheld from the liquid input entirely and
released later, on whatever day SWE actually declines, via `SM_j`.

**Direct, causal tension with the storage barrier, confirmed not just
inferred**: retraining with the phase-partition fix resolved a
genuine, pre-existing bug at Rustlers (a large winter precipitation
event was being misrouted as instant liquid flow — `Q_NSE` improved
-87.7 → -5.31, the single largest gauge-level fit improvement in this
project) — but **worsened chloride spikes materially at Copper and EAQ**
(Copper: 0 → 14 spike days; EAQ: 5 → 64). Reproduced across a second
random seed (confirming a real cost, not training-run luck), then
**tested directly with a causal intervention, not just correlation**:
took the trained checkpoint, forward-simulated it under the
counterfactual *un-withheld* forcing with no retraining — same weights,
different input. `S_s` on the affected days jumped 7.5x (Copper) to 17x
(EAQ), and the great majority of the new spikes vanished (Copper 13/14,
EAQ 64/64). **Confirmed mechanism**: withholding winter precipitation
removes water that used to prop `S_s` up through the winter, so it runs
lower more often, reactivating the same low-storage evapoconcentration
mechanism the barrier was built to suppress in the first place. Two
fixes that are each individually correct are in direct, demonstrated
tension with each other under the current architecture (see Section 2's
barrier re-sweep for the full characterization of why this can't be
resolved by re-tuning either one).

---

## 5. Rustlers' Aug-Nov chloride spikes: the false leads, and the final unification

Rustlers showed its own chloride spike pattern (Aug-Nov, both years) that
took several rounds to correctly explain — a useful record of what
*didn't* pan out, in a project with a pattern of plausible-looking
connections turning out to be separate mechanisms once tested:

- **Not caused by the barrier setting**: a full weight×threshold
  re-sweep (Section 2) showed Rustlers' spike count staying in the
  56-70 day range across the *entire* grid — no setting moves it
  materially.
- **Not caused by melt-timing release**: checked directly on the actual
  worst-residual dates (not just structurally) — `SM_j(t)=0` and
  `is_snow_day=False` on all 15 of the worst chloride-residual days.
  Definitively ruled out.
- **Not caused by an "oversized melt burst"** from withheld precipitation
  being released later: `SM_j(t)`'s formula never changed between the
  snowmelt-only and phase-partition rounds (diffed directly — byte-
  identical, max difference 0.0), so there is no code path where
  withheld water reappears amplified.
- **A real, separate, previously-unflagged numerical issue was found and
  ruled out as the cause**: `C_f` (fast-reservoir concentration) has no
  ceiling anywhere in the code (unlike `C_tot`), and reached
  2,044,598 mg/L — ~6.7 million times the learned end-member — when `S_f`
  crashes toward its floor with no protection analogous to the barrier
  on `S_s`. But checked directly: `corr(|residual|, C_f) = -0.123` (weak,
  wrong sign) — `Q_f` collapses simultaneously, self-limiting `C_f`'s
  actual effect on `C_tot`. Confirmed not the driver of Rustlers'
  spikes, but flagged and later fixed as independent code hygiene
  (Section 9's changelog).

**The final, correct answer**: a full state snapshot on the worst
residual days found the dominant signal was `C_s` itself
(`corr(|residual|, C_s) = 0.856`, the cleanest correlation found in this
entire investigation) — but the raw *level* of `S_s` was a much noisier
predictor at Rustlers than it had been at Copper/EAQ. The deciding
check: does `C_s`'s correlation trace back to `S_s` depleting while `M_s`
(tracer mass) stays passively stable — the same causal structure as
Copper/EAQ — or is `M_s` itself doing something independently unusual?
On the worst-residual days, mean `S_s` is 0.58x the overall mean (real
depletion) while mean `M_s` is 1.07x the overall mean (essentially
unchanged) — `M_s` is not spiking or crashing, it simply fails to keep
pace as `S_s` shrinks. `corr(S_s, M_s) = 0.925` across the full record
confirms they normally move in lockstep. **Confirmed: this is the same
low-storage evapoconcentration mechanism as Copper/EAQ, not a separate,
unexplained one** — it triggers at a shallower relative depletion
(~58% of Rustlers' mean vs. ~7-9% of Copper/EAQ's mean) simply because
Rustlers is a smaller, more sparsely-observed gauge where moderate
storage depletion is already common, making the raw `S_s`-level
correlation noisier even though the underlying cause is identical.
Practically still unresolved (the barrier sweep already showed no
setting fixes it), but now correctly attributed rather than filed as its
own separate, open mystery.

---

## 6. PH_LT peak: timing fixed, magnitude/shape a characterized data limitation

PH_LT's snowmelt peak was wrong in every round of this project — this
was the one open thread that got a dedicated, systematic investigation
late in the project, in the same style as baseflow fraction.

**Precise characterization first, before any fix**: 2014's simulated
peak landed 13 days early (0.93x observed magnitude); 2015's landed 11
days early and badly undershot (0.38x). Cross-correlation over a
widened window jumped from weak (0.36-0.40 at zero lag) to strong
(0.79-0.85) once the simulated series was shifted ~14-18 days later —
**timing was the dominant component of the mismatch**. But real,
year-inconsistent shape differences remained underneath that: 2014's
peak was too narrow (a fast recession-limb problem), 2015's too broad —
opposite directions, so no single "widen/narrow" fix could address both.

**Peak-weighted loss, tested and rejected**: the natural hypothesis
(the loss under-weights rare high-flow days) was tested directly, not
assumed — computed what fraction of each gauge's loss its own top-10
highest-observed-discharge days already command. At PH_LT, those 10 days
(1.6% of its observations) already contribute **27.6%** of its entire
loss — a 16.86x over-representation. **The opposite of what the
hypothesis needs**: peak days are already substantially over-weighted by
ordinary squared-error loss, because that's where the model's errors are
largest. Reweighting was not implemented; the cause lies elsewhere.

**Melt-timing lag, tested cheaply first, then implemented**: a
no-retraining forward-pass test — shifting the existing `SM_j(t)` signal
forward by a range of offsets and re-simulating through the trained
network with no other change — found a shared ~10-14 day lag brought
both years' timing close to zero simultaneously, while magnitude stayed
essentially flat regardless of offset (2015's ratio: 0.37-0.39 at every
single offset tested, including the timing-optimal one). This
distinguished the two problems cleanly *before* committing to a retrain.
Implemented properly (`MELT_LAG_DAYS=10` in `data.py`, both variants
retrained): **timing is now fixed** — both years land within 2 days of
the observed peak date. **Magnitude did not improve**, exactly as
predicted: 2014 now overshoots (1.31x, aligning the timing moved the
peak onto a day where the model's magnitude was already running high);
2015 stays flat (0.36x).

**Why magnitude/shape is treated as a closed, accepted limitation, not
an open bug**: both testable model-side hypotheses (loss weighting,
timing) were tested to completion and either rejected or confirmed
partial. What remains is attributed to the single-station SNOTEL proxy's
inability to supply the correct total melt *mass* for sub-basins it
doesn't measure — the same class of limitation the formulation doc
already documents for the ET/CLM forcing (Rustlers and EAQ flagged as
>3.5km from their nearest matched meteorological station). A single
point can capture *when* melt happens reasonably well (confirmed: one
constant lag correction works for both years) but there's no mechanism
by which it can supply the correct *magnitude* for basins it doesn't
actually measure. Fixing this would require a per-sub-basin SWE product
or a distributed snow model, neither of which exists in this repo. This
is documented as characterized and closed pending better data, not a
standing item to keep re-investigating.

---

## 7. Gothic_ME's catastrophic NSE: pre-existing, structural, unrelated to any fix tested

Gothic_ME's discharge `Q_NSE` has been catastrophically negative
(three-to-four digits, e.g. -1025 to -2784 depending on the round) in
**every single checkpoint trained in this entire project, including the
very first one** (-2027.5). This was initially mischaracterized as a
"new" problem introduced by the snowmelt term; closer inspection showed
the snowmelt round made it worse (roughly 2.7x relative to the
immediately preceding checkpoint) but did not create the underlying
tendency.

**The mechanism, characterized directly**: Gothic_ME's residual
correlates moderately with `SM_j(t)` (r≈0.48-0.51) — real, but the same
spurious-surge tendency was already present, at smaller magnitude,
*before* the snowmelt term existed (the pre-snowmelt checkpoint already
produced a ~2.5 cms spurious burst in the same May-2015 window against
~0.03 cms observed, plus other spurious peaks entirely unrelated to melt
season, e.g. a February burst). **Why Gothic_ME specifically**: it has
the largest local contributing area of any modeled node (26.08 km²) and
one of the smallest natural observed-flow scales (max ever observed 0.49
cms) — a combination that turns any spurious input-driven volumetric
burst into both a large absolute value (large area) and a catastrophic
NSE (NSE's variance-normalized denominator is tiny for a gauge that
barely moves). This is **not** explained by the documented
station-distance limitation (the formulation doc flags Rustlers and EAQ
for poor meteorological-station matching, not Gothic_ME) — the pattern
fits area×flow-scale, not spatial representativeness, and EAQ (which
*is* flagged) actually improved to a positive `Q_NSE` in the same round
Gothic_ME got worse.

This remains open, structural, and — per the same discipline applied to
Rustlers and PH_LT — is not something any fix tried in this project
(barrier tuning, melt term, phase partitioning, melt-timing lag) has
touched or was expected to touch, since none of them target basin-area
scaling or NSE's sensitivity at near-zero-flow gauges.

---

## 8. What was deliberately not done, and why

One significant intervention was identified early, held in reserve, and
**never implemented** at the time this section was first written — by
decision, not oversight. A second was flagged the same way and has since
been implemented and tested; see the update below.

- **~~A structural minimum-recharge-fraction constraint~~ — UPDATE: this
  has now been implemented and tested, see Section 12.** (Originally
  described here as "architecturally guaranteeing some fraction of
  infiltration always reaches the slow reservoir" — that description
  itself turned out to be imprecise, corrected in Section 12: the
  model's freely-learned recharge fraction, 0.65-0.9, already exceeds
  any literal *minimum* near 0.3, so what actually needed implementing,
  per Section 3's own forced-sweep finding, was a *ceiling*, not a
  minimum.) Flagged as an option as early as the storage-barrier round,
  explicitly held in reserve while two more surgical fixes (the barrier,
  the `C_tot` ceiling) were tried first. Once the recharge-fraction
  forced sweep (Section 3) confirmed recharge fraction as a real, strong
  lever and the loss-surface check confirmed the *data loss itself*
  actively fights any move toward the target range, this became the
  more clearly-indicated next step for baseflow fraction specifically —
  but by that point the investigation had moved to testing *why* the
  loss prefers high recharge fraction (the `K_f`/`K_s` recession-mismatch
  finding), and then to the snowmelt-term chain, which moved baseflow
  fraction substantially through a different mechanism instead, and it
  stayed untried for the remainder of the original investigation.
  Implemented and tested in a later round (Section 12): a soft ceiling
  moves recharge fraction and baseflow fraction substantially in the
  right direction, cheaply, but does not on its own reach the benchmark
  band at the tested weight.
- **Per-sub-basin SWE disaggregation** (replacing the single Butte 380
  station, applied uniformly to all 5 sub-basins, with a spatially
  distributed snow product or model). Identified as the leading
  candidate explanation for both PH_LT's remaining peak magnitude gap
  and Gothic_ME's melt-correlated component, in both cases **because no
  such data exists in this repo** — not a modeling choice but a hard
  data-availability constraint. Documented explicitly as the fix that
  would be needed, rather than silently left unaddressed.
- **Per-gauge barrier calibration** (a 5x-parameter generalization of
  the single global barrier weight/threshold) was actively considered,
  partially budgeted for, and then **ruled out with direct evidence**
  before being attempted at scale: a cheap check (using the
  already-completed global sweep, no new training) found all 5 gauges
  agree on the same preferred setting for both contested metrics — there
  is no gauge-specific divergence for a per-gauge dial to exploit, so
  the added complexity would not have helped. This is different from the
  other two items: it was not merely deferred, it was actively
  investigated and found not to be worth pursuing.
- **Peak-weighted loss** (Section 6) was tested and rejected with direct
  evidence, not left untried.

---

## 9. Current best checkpoint: recommendation and honest status

**Recommended checkpoint**: `results_meltlag_single` / `results_meltlag_split`
— phase-partitioned `P_e` (Section 4), the 10-day melt-timing-lag
correction (Section 6), and the storage barrier and `C_tot` ceiling at
their established settings (weight=60, threshold=1.0mm — the barrier
sweep's own baseline, since no alternative setting in either sweep
dominated on all fronts) — **served through the current `model.py`,
which includes a defensive `C_f` ceiling in code** (`CF_CEILING_MGL`,
derived the same way as `CTOT_CEILING_MGL`: the highest chloride
concentration ever recorded anywhere in this project's data — the
ER-PLM8 groundwater well, 2617.49 umol/L = 92.80 mg/L — times a 5x
safety margin). **This is a deliberate reversal of an intermediate
recommendation** (`results_cfceiling_single/split`, a full retrain with
the ceiling baked in): the retrain was tested directly against applying
the same ceiling to these same, already-trained weights with no
retraining at all, and the retrain measurably *lost* ground on the
project's central open metric (baseflow fraction, worse at 4 of 5
gauges) and on EAQ's chloride fit (peak concentration more than 3.5x
higher) for a benefit — bounding a value already shown not to cause any
active problem — that the no-retrain option captured just as well, with
none of the cost. `results_cfceiling_single/split` remain in the repo,
inspectable, as the record of that (correctly rejected) experiment. Both
`single_C_in` and `split_C_in` variants are maintained; `single_C_in`
has the more thoroughly cross-checked diagnostic history in this project
and is the safer default if only one is needed.

**What it gets right**:
- **Floor-crashing**: solved and robust at 4 of 5 gauges (0.0-1.2%).
  EAQ remains a persistent exception (2.4-5.4%) — confirmed connected to
  genuine multi-day zero-input droughts in the phase-partitioned forcing
  (Section 2), not fixable by barrier re-tuning.
- **Chloride spikes at most gauges**: outlet (PH_LT) chloride is clean —
  5-7 spike days, magnitude just above the 1 mg/L threshold (max
  1.13-1.20 mg/L), a large reduction from the original ~60 mg/L
  excursions. Gothic_ME similarly clean (8-10 days).
- **PH_LT peak timing**: fixed. Both 2014 and 2015 now land within 2
  days of the observed peak date, down from 11-13 days early.
- **Rustlers' pre-existing winter-precipitation bug**: resolved.
  `Q_NSE` improved from -87.7 (phase-partition round) to -6.1/-7.2 on
  this checkpoint — down from -262.9 in the original, pre-barrier
  checkpoint — the single largest gauge-level fit improvement achieved
  by any fix in this project.
- **The ODE integration, routing topology, and split-`C_in` physical
  constraints**: correct and exact by construction throughout, never in
  question at any point in this investigation.
- **`C_f` is bounded in code** (`CF_CEILING_MGL≈464 mg/L`, closing a
  latent numerical-stability gap — unprotected `C_f` previously reached
  2,044,598 mg/L in one checkpoint) **without the cost of retraining on
  it**: applying the ceiling to these already-trained weights at
  inference only *strictly improves* chloride spike-day counts at every
  gauge (single-C_in: PH_LT 9→7d, Rustlers 74→64d, Copper 28→21d, EAQ
  69→40d, Gothic_ME 15→10d) with zero effect on floor-crash, `Q_NSE`, or
  baseflow fraction (confirmed — a frozen forward pass cannot change the
  water balance). See Section 10 for why retraining on this same change
  produces different, worse results, and `pinn/README.md`'s
  "Recommendation reversed" section for the full three-way comparison.

**What remains open, stated plainly**:
- **Baseflow fraction is still above the literature benchmark at every
  gauge** (72-83% annual on this checkpoint vs. a 12-33% target) — the
  one original target metric still clearly unresolved, despite being the
  most-investigated single question in this project. The mechanism is
  well understood (recharge fraction, confirmed as a real, strong lever
  whose barrier is the data loss's own preference, not initialization).
  **Update**: the recharge-fraction ceiling constraint (Section 8/12) has
  since been implemented and tested — a soft version moves baseflow
  fraction substantially in the right direction at every gauge but does
  not reach the benchmark band at the tested weight; a higher weight or a
  hard constraint remains the indicated next step if this is prioritized
  further, at a data-loss cost that Section 3's loss-surface measurement
  suggests will scale up as the constraint is tightened.
- **Isotope (d-excess) cross-validation has never shown a strengthened
  signal in any round** — correlations stay weak (|r| typically <0.3)
  and inconsistent in sign across gauges throughout the entire project.
  Not disconfirming, but never confirmatory either; treat the model's
  fast/slow partitioning as statistically under-supported by the isotope
  data as currently used.
- **PH_LT's peak magnitude/shape**: characterized and closed as a
  single-station-SNOTEL data limitation (Section 6), not expected to
  improve without a per-sub-basin SWE product.
- **Gothic_ME's discharge NSE**: structural, pre-existing, unrelated to
  any fix tried (Section 7) — a known limitation of NSE at a
  large-area/near-zero-flow gauge, not a target for further tuning.
- **Rustlers' Aug-Nov chloride spikes and Copper/EAQ's chloride
  regression from the phase-partition fix**: both confirmed as the same
  known low-storage evapoconcentration mechanism (Sections 4-5), and
  both confirmed **not fixable by any barrier setting** — an
  architectural tension between two individually-correct fixes, open
  pending either a different storage-protection mechanism or acceptance
  of the trade-off.

**Net assessment**: this project fixed every problem it set out to fix
via disciplined, hypothesis-then-direct-test iteration — floor-crashing,
unbounded concentration, the missing-snowmelt structural cap, and PH_LT's
peak timing are all solved with strong, reproducible, causally-verified
evidence. The literature baseflow-fraction benchmark remains unmet, not
for lack of investigation but because every mechanism found to move it
(recharge fraction, the snowmelt term) also moves something else the
wrong way, and the specific fix positioned to resolve it directly (a
structural recharge-fraction constraint) has not yet been tried. And the
project's own final round is itself a data point worth keeping in view:
a change that is confirmed non-cosmetic is not automatically worth
retraining on — see Section 10.

## 10. General finding: loss terms share parameters, so a fix to one can move the other — check, don't assume

The `C_f`-ceiling episode (Sections 4 and 9) is worth stating as its own
general property of this architecture, not just as an explanation local
to that one fix. **`K_f,j(t)`'s four bounds, `C_in,fast,j`
(and `C_in,gw,j` in the split variant), and the partition network's
weights are trained jointly from both the discharge loss and the
chloride loss** — there is no separation between "the parameters that
determine water balance" and "the parameters that determine chloride."
Any change to how the chloride loss computes or weights its terms
changes the gradient those shared parameters receive, and since the
*same* parameters also set `S_f`, `S_s`, `Q_f`, `Q_s`, and `Q_tot`, that
change can alter the trained water-balance behavior — even though the
water-balance *forward equations* never reference chloride at all, and
even when the change (like the `C_f` ceiling) is applied to an
intermediate variable with no direct physical influence on discharge.

This was confirmed directly, not inferred: applying the `C_f` ceiling to
already-trained weights at inference time left floor-crash rate,
`Q_NSE`, and baseflow fraction completely unchanged (a frozen forward
pass genuinely can't propagate a downstream clamp backward into water
balance) — but retraining *with* the same ceiling in place from scratch
produced measurably different values for all of them, and a determinism
test (identical seed and code, rerun) proved this wasn't training noise:
it was the ceiling's altered gradient signal reaching shared parameters
during backpropagation and changing what the optimizer did with them
over 1000 epochs.

**The practical implication for any future change to this
architecture**: a modification framed as "chloride-only" or
"discharge-only" is only guaranteed to be metric-only if it's applied to
frozen, already-trained weights. The moment it's included in a *retrain*
from scratch (or a continuation of training), it should be assumed to
have some effect on the other loss's target metrics too, however
indirect the physical path — and that effect should be checked
empirically (as here) rather than reasoned away from the forward
equations alone, which was exactly the assumption ("chloride can't
affect the water balance") that turned out to be true only at the
level of physics, not at the level of training dynamics.

**Correction, tested rather than left as an assumption**: an earlier
version of this section implied inference-only application might
generally be worth trying before committing to a retrain, since it
worked cleanly for the `C_f` ceiling. Tested directly on the other two
major fixes in this project (same style: apply the fix to the pre-fix
checkpoint's weights at inference only, compare against the actual
retrain) — **it does not generalize**:

- **Phase-partitioning**: a genuine trade-off, not a dominance either
  way. Inference-only application (pre-fix weights, phase-partitioned
  `P_e` as input, no retraining) wins baseflow fraction at all 5 gauges
  and most chloride/`Q_NSE` metrics — but loses badly at PH_LT (`Q_NSE`
  flips from +0.44 under the retrain to -0.07 without it) and at
  Gothic_ME. Neither option is simply better.
- **Storage barrier**: inference-only **fails outright**. A hard `S_s`
  floor at the barrier's own threshold, applied to the pre-barrier
  weights with no retraining, left Rustlers' floor-crash rate completely
  unchanged (15.3% → 15.3%) and made EAQ's slightly worse (14.7% →
  15.2%) — while the actual retrain solved floor-crashing almost
  entirely (→ 0.3% or better) and substantially improved `Q_NSE`
  everywhere. The barrier doesn't work by pinning `S_s` at a floor; it
  works by shifting gradient descent's entire trajectory so `S_s`
  settles at a much higher **emergent equilibrium** (37-83mm, 40-80x the
  threshold itself) — something only training, not a forward-pass clamp,
  can produce.

**The corrected general lesson**: whether inference-time application can
substitute for retraining depends on *what mechanism the fix relies on*,
not on whether it's confirmed non-cosmetic. A fix that bounds a value
which is only rarely extreme (`C_f`'s ceiling) can be fully captured at
inference, because the trained dynamics barely need to change to comply
with it. A fix that requires the network to have learned different
*equilibrium* behavior (the storage barrier) cannot be substituted at
all — training is the mechanism, not incidental to it. A fix that
changes the *input data* itself (phase-partitioning) produces a real,
substantial effect at inference, but not one that's strictly better than
retraining — a genuine trade-off requiring the same kind of direct
before/after evaluation as any other fix, not a shortcut. "Check
empirically" remains the operative instruction from this section; "try
inference-time first as a general strategy" does not follow from it, and
saying so without testing it against fixes other than `C_f`'s would have
been exactly the kind of unverified generalization this project has
consistently found reasons not to make.

## 11. The 3-way comparative analysis: does physics help, does the tracer help — tested against an LSTM, on genuinely held-out data

### 11.1 Why a held-out split was required, and what it cost

Every prior section evaluates the tracer-informed PINN in isolation, and
every prior checkpoint (including the one recommended in Section 9) was
fit against **100% of the observed record with no held-out split** — a
legitimate way to characterize one model's own behavior, but not a valid
basis for asking whether physics or the tracer actually help relative to
an unconstrained baseline. Scoring one model in-sample and comparing it to
another model's out-of-sample score biases the comparison mechanically, in
the PINN's favor, for reasons that have nothing to do with which model is
actually better. **Concretely, this meant the existing Section-9 checkpoint
could not be reused for this comparison at all** — it was fit on 100% of
the data, so its metrics on any subset of that same data are in-sample by
construction, not held-out. The tracer-informed PINN had to be retrained
from scratch under a genuine split before any comparison against the LSTM
could mean anything.

No train/test split existed anywhere in this codebase before this round
(`pinn/split.py`). A chronological cutoff was checked first and rejected:
Rustlers and Copper have zero discharge observations before mid-2015, so
any single date split leaves those two gauges with no training signal at
all. Instead: a fixed-seed (42), stratified random 80/20 split of each
gauge's own observed days, done independently for Q and Cl, shared
identically across all three models below (loaded from one saved
`results_comparative/holdout_split.npz`, not regenerated per model — exact
train/test observation counts per gauge, logged in every checkpoint's
`predictions.npz`, are identical across all three by construction).
Forcing (`P_e`, `ET`) is never masked — only which *observed* days feed
each model's training loss vs. its held-out evaluation changes. Per-gauge
normalization statistics used inside the loss are computed from the train
split only, to avoid leaking the held-out distribution into training even
indirectly.

**Three models, same held-out split:**
1. **Tracer-informed PINN** (`results_comparative_tracer_single`) —
   `HydroPINN(split_cin=False)`, retrained from scratch under the split.
2. **Hydrology-only PINN** (`results_comparative_hydro_only`) — identical
   architecture, `--cl-weight 0.0`. No `model.py` change was needed: the
   water-balance forward equations never read tracer state, so zeroing
   the loss term is sufficient. One real consequence: `Cin_fast`/`Cin_gw`
   then receive zero gradient from anywhere and stay at initialization, so
   this variant's `Cl` output is untrained and not reported anywhere below
   (reporting it would be a fabricated number, not a real one).
3. **LSTM baseline** (`results_comparative_lstm`, `pinn/train_lstm_baseline.py`)
   — a single joint LSTM over all 5 sub-basins' forcing at once (not 5
   independent per-basin models — Gothic_ME/PH_LT's observed discharge is
   a *routed* total that a basin-local model has no way to see), 1 layer,
   hidden=64, ~20,618 parameters, `[P_e,1..5(t), ET,1..5(t), sin(doy),
   cos(doy)]` input, softplus output head (5 Q + 5 Cl), trained with the
   same masked, per-gauge-normalized loss convention as the PINN scripts.

**Sanity check before trusting the retrain**: the tracer PINN's held-out
retrain reproduces Gothic_ME's catastrophic `Q_NSE` (train=-1330,
test=-2133) at the same order of magnitude as the original 100%-data
Section-9 checkpoint (`Q_NSE=-1378`) — confirming Section 7's "pre-existing,
structural" characterization again, on an independently retrained model,
and confirming this round's training pipeline isn't itself broken (a
badly wired loss/mask would likely produce a *different* pathology, not
reproduce the exact same one).

### 11.2 Headline result: the LSTM wins on raw held-out accuracy, everywhere

`figures_final/comparative_analysis/01_nse_comparison_holdout.png`. The
LSTM baseline beats **both** PINN variants on `Q_NSE`, on held-out data,
at **every single one of the 5 gauges** (Rustlers: LSTM 0.78 vs. tracer
-22.17 / hydrology-only -6.15; Copper: 0.97 vs. -0.34 / -0.20; EAQ: 0.85
vs. -0.08 / -0.16; Gothic_ME: 0.68 vs. -2133.43 / -2346.15; PH_LT: 0.97
vs. 0.37 / 0.32), and also beats the tracer PINN on `Cl_NSE` at every
gauge (Rustlers: -0.21 vs. -13.82; Copper: -0.62 vs. -1.04; EAQ: -0.59 vs.
-13.10; Gothic_ME: -1.58 vs. -3.72; PH_LT: +0.64 vs. -0.42). **Stated
plainly: on raw held-out prediction accuracy, in this setup, neither
adding physics nor adding the tracer improved on an unconstrained LSTM at
any of the 5 gauges, for either variable.**

### 11.3 Why that's not the whole story: the chloride "win" is overfitting, confirmed data-scarcity-driven and chloride-specific

This is not the whole picture, and the gap is itself informative, not
just a scoreboard. Comparing each model's own train vs. held-out-test NSE
shows *why* the LSTM wins on raw accuracy, and for chloride it is a
classic overfitting signature, not uniformly better generalization. LSTM
train `Q_NSE` is 0.98-0.995 at every gauge, dropping to 0.68-0.97
held-out — a real but modest generalization gap. `Cl_NSE` is a different
story: LSTM train fit is 0.87-0.9995 (near-perfect memorization of 38-460
sparse points) collapsing to **negative** held-out `Cl_NSE` at 4 of 5
gauges (only PH_LT, with 460 training days, holds up: 0.64 held-out). The
tracer PINN's train-vs-test gap is much smaller in absolute NSE units at
most gauges, but for a less flattering reason than "it generalizes
better": its *training-set* `Cl_NSE` is already poor (-14.26 to -0.40), so
there is little fit left to lose. The PINN's physical constraints visibly
prevent the kind of severe memorization the unconstrained LSTM exhibits on
this small, sparse dataset — but at the cost of a training-set fit that
never gets good in the first place, and a held-out accuracy the LSTM still
exceeds despite its overfitting. Both things are true simultaneously;
neither cancels the other.

**Is that chloride overfitting about this LSTM's size, or about data
scarcity?** (`05_lstm_capacity_ablation.png`.) Tested directly, not
assumed: a second LSTM, hidden=16 (2,090 params, ~10x fewer, identical
architecture otherwise), trained from scratch under the *identical*
held-out split. **The failure persists in kind at all 4 sparse-Cl gauges
even at 1/10th the parameters** — held-out `Cl_NSE` stays negative at
Rustlers (-0.21 → -0.28), Copper (-0.62 → -0.20), EAQ (-0.59 → -0.34), and
Gothic_ME (-1.58 → -0.22) at both capacities, and only PH_LT (460 training
days, an order of magnitude more than the other four's 38-43) is positive
at both (0.64 → 0.60). The train-test gap does shrink with lower capacity
(mean gap across the 4 sparse gauges: 1.75 at hidden=64 → 1.20 at
hidden=16 — capacity is not irrelevant to *how severely* the model
memorizes), but the qualitative outcome — can this architecture achieve
positive held-out `Cl_NSE` from ~40 observations — does not change at
either size tested.

**The negative control that pins this down as chloride-specific, not a
general architecture limitation**: discharge, which is not comparably
sparse at any gauge (59-488 training observations vs. chloride's 38-460),
shows no equivalent pattern — held-out `Q_NSE` at hidden=16 is comparable
to, and in 3 of 5 cases better than, hidden=64's. If the LSTM's held-out
weakness were a general property of the architecture (too flexible, too
easily thrown off by a smaller network, whatever the mechanism), discharge
should degrade in the same way chloride does when capacity drops by 10x.
It doesn't. **This is the stronger, more general form of the finding**:
the chloride failure is not "this particular LSTM was oversized," it's
that a black-box sequence model with no informative prior needs materially
more chloride observations than 4 of these 5 gauges have to generalize at
all — a data-scarcity-driven limit specific to the sparsely-observed
target variable, not a tunable hyperparameter problem, at least across the
one order of magnitude of capacity tested here.

### 11.4 What only the PINN variants can be evaluated against at all

`02_baseflow_fraction_comparison.png`, `03_isotope_crossval_comparison.png`.
Baseflow fraction against the Hubbard et al. benchmark (not independently
re-verified here), and fast-fraction correlation against isotope d-excess, both
require an internal fast/slow reservoir state the LSTM does not have — not
"not computed for the LSTM" but **structurally inapplicable to it by
construction**: there is no fast/slow decomposition anywhere in an LSTM's
architecture for either diagnostic to be computed from. Both PINN variants
land at 70-90% mean annual baseflow fraction at every gauge, well above
the 12-33% benchmark band regardless of whether the tracer was trained
(tracer and hydrology-only track each other closely at every gauge —
consistent with Section 3's finding that this is a loss-surface/
architecture property, not something the tracer loss moves), and isotope
cross-validation correlation is weak and similarly near-zero for both
variants at every gauge (r between -0.23 and -0.01) — the tracer loss does
not measurably change either diagnostic relative to the hydrology-only
variant. Only the two PINN variants expose any inspectable physical
parameter (`K_f,j(t)`'s bounds, `K_s`, `alpha_ET`, and — tracer-informed
only — `C_in,fast/gw`); the LSTM is a black box by construction, with no
parameter that corresponds to a named physical quantity.

### 11.5 Bottom line

This comparison does not support a claim that physics or the tracer
improve raw predictive accuracy over a simple black-box baseline on this
dataset — the opposite is true, consistently, at every gauge tested, and
that result survived a genuinely held-out evaluation, not an in-sample
one (Section 11.2). Part of the LSTM's apparent advantage is itself an
artifact worth naming precisely rather than either dismissing or ignoring:
on chloride specifically, it is overfitting to a handful of sparse
observations, confirmed both by the train/test gap and by a 10x capacity
ablation that leaves the failure unchanged at the 4 data-poor gauges while
leaving discharge — not comparably sparse — unaffected (Section 11.3).
That confirms the chloride result is a genuine data-scarcity limit of the
unconstrained approach, not a tuning artifact of one LSTM configuration —
but it does not overturn the headline: even accounting for it, the LSTM's
held-out accuracy still exceeds the PINN's at every gauge, including
discharge, where sparsity is not the explanation.

What physics buys instead, in this project, is what Sections 1-10 are
actually about, and what Section 11.4 makes concrete by contrast: a small
number of interpretable parameters, and a structural fast/slow
decomposition that can be checked against independent benchmarks and
tracers the model was never fit to — checks that are not just unreported
for the LSTM but have no analog to report, because nothing in its
architecture corresponds to the physical quantities those checks require.
**Physics and the tracer did not win on raw predictive accuracy in this
test. What they bought instead was the only output in this comparison
that could be checked against anything outside the training data itself.**
Whether that trade is worth it depends entirely on what the model is for;
this section reports the trade, not a verdict on it.

### 11.6 Implementation notes (compute cost, which LSTM run)

For scale: the LSTM (20,618 params) trained in 56s on this environment's
CPUs; each PINN variant (1,475 params — two orders of magnitude fewer)
took ~2000-2040s (~34 min) for the same 1000 epochs, because its
architecture requires a sequential 737-step Python-level loop (the exact
closed-form ODE update, once per day, per the class docstring in
`model.py`) rather than a single vectorized recurrent-kernel call.

**Explicit confirmation on which LSTM run produced the numbers in 11.2 and
11.3**: the LSTM was first launched concurrently with both PINN jobs and,
due to severe CPU thread-pool contention between three simultaneously-
running multi-threaded PyTorch processes, ran for over an hour without
finishing (confirmed as contention, not a hang: `ps` showed it still
actively consuming CPU throughout, and an isolated timing test of the
identical script/arguments — no other job running — completed 3000 epochs
in 55.79s). That contended run was killed (`kill -9`) before it ever wrote
`metrics.json`, `model.pt`, or any other output file —
`results_comparative_lstm/` was completely empty (`total 8`, directory
entries only) at the moment of the kill. The LSTM was then re-run alone,
once both PINN jobs had already finished, and completed in 55.79s
(recorded in `results_comparative_lstm/metrics.json`'s own
`train_config.train_time_s` field). **Every number in this section, and in
every figure in `figures_final/comparative_analysis/`, comes from that
clean 56-second run** — the killed 68-minute run never produced any
artifact that could have been used, so there was no actual ambiguity, only
an implicit claim worth stating this plainly.

### 11.7 Multi-seed robustness: does the LSTM win at every gauge across every seed?

Sections 11.2-11.3 report a single seed (0) per model. Retrained the LSTM
across 5 seeds (0-4, ~1 min each) and the tracer-informed and
hydrology-only PINN across 3 seeds each (0-2, ~34 min each) — a
compute-driven, explicitly disclosed scope reduction from 5 seeds:
each additional PINN seed costs roughly 34x an LSTM seed on this
environment's CPUs, and 5 PINN seeds per variant (10 additional runs at
~34 min each, ~5.7 hours) was judged not affordable this round; 2
additional seeds per PINN variant (4 runs, ~2.3 hours) was. All seeds
under the identical held-out split (`figures_final/comparative_analysis/06_multiseed_robustness.png`).

**Discharge: the headline result is fully robust.** The LSTM's held-out
`Q_NSE` beats both PINN variants' full seed range at every one of the 5
gauges — even the LSTM's *worst* seed at every gauge outperforms both
PINN variants' *best* seed there. Confirmed, not assumed.

**Chloride: the headline result holds at 4 of 5 gauges, not all 5.** At
Rustlers, Copper, EAQ, and PH_LT, the LSTM's worst-seed `Cl_NSE` still
beats the tracer PINN's best-seed `Cl_NSE`. **At Gothic_ME, it does not**:
tracer PINN `Cl_NSE` is tightly clustered (-3.76 to -3.53 across 3 seeds),
while the LSTM's is highly volatile there (-5.08 to -1.30 across 5 seeds)
— the LSTM's worst seed (-5.08) is worse than the tracer PINN's best seed
(-3.53). This is not the tracer PINN improving at Gothic_ME; it is the
LSTM itself being unstable there. **Stated precisely**: the "LSTM wins everywhere" framing from Section 11.2 is correct
for discharge at all 5 gauges and for chloride at 4 of 5 gauges, across
every seed tested — but at Gothic_ME specifically, the LSTM's own
seed-to-seed chloride volatility is large enough that a single unlucky
seed can lose to the tracer PINN's typical performance there. This
doesn't change the mean-based comparison in Section 11.2 (the LSTM's
mean `Cl_NSE` at Gothic_ME, -2.83, is still better than the tracer PINN's
mean, -3.67) — it changes only the strict "wins on every single seed"
claim, which is a stronger, different claim than "wins on average."

**Follow-up check, apples-to-apples**: is the tracer PINN's tight
Gothic_ME chloride clustering itself just a small-sample artifact, or is
it real? Checked directly against its own 3 seeds individually: -3.72
(seed 0), -3.53 (seed 1), -3.76 (seed 2) — range width 0.22, vs. the
LSTM's 5-seed range width of 3.78 there (17x wider). The clustering is
real, not a lucky draw. The hydrology-only PINN has **no** seed-to-seed
chloride variance to report at Gothic_ME or anywhere — confirmed
directly in every seed's `metrics.json` (`Cl_NSE_test: None`, `Cin_fast`/
`Cin_gw` never received gradient) — there's no trained output to have
variance in, not a small-variance one; stated plainly rather than forcing
a three-way comparison that only has two real members for chloride.

### 11.8 Does split_C_in narrow the gap to the LSTM?

The 3-way comparison (11.1-11.7) used only `single_C_in` as "the"
tracer-informed PINN, a deliberate choice for a clean complexity ladder
(11.1). Retrained `split_C_in` (two chloride end-members) from scratch
under the identical held-out split (`results_comparative_tracer_split`)
to test directly whether the more physically-motivated formulation closes
any of the gap to the LSTM
(`figures_final/comparative_analysis/07_split_cin_comparison.png`).

**No, not consistently — the result is genuinely mixed, with one sizable
regression.** `split_C_in` narrows the gap to the LSTM at only 2 of 5
gauges for discharge and 2 of 5 for chloride — not a majority either way.
Where it helps, the improvement is real: Rustlers' `Q_NSE` improves from
-22.17 to -14.29 and Copper's crosses from negative to positive (-0.34 ->
+0.10); Rustlers' and EAQ's `Cl_NSE` both improve substantially (-13.82 ->
-13.07 and -13.10 -> -9.13). But **Gothic_ME's `Q_NSE` gets dramatically
worse under `split_C_in`** -- -2133.43 (single) -> -6615.95 (split), a
3.1x deterioration on an already-catastrophic, structurally-understood
metric (Section 7) -- and Copper's and PH_LT's `Cl_NSE` both degrade
(-1.04 -> -3.46; -0.42 -> -0.88). This does not match either of the two
hypotheses posed going in ("narrows the gap" or "tracks single_C_in
closely") cleanly: it is neither a consistent improvement nor a close
tracking -- it is gauge-specific, sometimes helping substantially,
sometimes hurting substantially, with no discernible pattern tying the
direction to gauge characteristics (sparse vs. dense chloride data, or
headwater vs. routed position) that would let it be predicted in advance.

**Follow-up check on the Gothic_ME `Q_NSE` regression specifically**: is
-6615.95 outside the noise this metric already shows at this gauge, or
within it -- checked directly rather than left as an unexplained "real
effect," consistent with how every other regression in this project has
been handled. Two comparison populations give different answers. Against
the pre-holdout, 100%-data-fit checkpoints spanning this whole project (32
checkpoints, range -1025.2 to -2918.1), -6615.95 does fall outside --
more than 2x beyond that range's floor. But that population was trained
under a different regime entirely (100% of the data, no held-out split).
The directly comparable population -- checkpoints trained under the
*identical* held-out-split regime, this same round -- tells a different
story: `single_C_in`'s own 3 seeds alone swing from -2133.43 (seed 0) to
-8992.67 and -10549.96 (seeds 1, 2); hydrology-only's swing from -2346.15
to -10247.66 and -11244.12. **`split_C_in`'s single observed value
(-6615.95) falls inside that already-directly-observed range -- milder
than 4 of the other 5 held-out-split multi-seed runs on record**, all
from the same architecture (`single_C_in`) or its hydrology-only sibling,
differing only in random seed. **Corrected conclusion**: -6615.95 is well
within the range of seed-to-seed noise already directly observed for
Gothic_ME `Q_NSE` under this exact training regime -- the 3.1x
"deterioration" is much more plausibly ordinary training-seed variance at
a metric Section 7 already documents as structurally hypersensitive at
this gauge, not a genuine `split_C_in`-specific effect. One honest limit:
only a single `split_C_in` seed was run, so its own seed-to-seed spread
isn't directly measured the way `single_C_in`'s and hydrology-only's are
(3 seeds each) -- this is a strong plausibility argument from the
regime's demonstrated volatility, not a multi-seed confirmation specific
to `split_C_in` itself.

**The two-end-member formulation does not reliably buy anything in the
held-out setting for the gauges where a real signal is measurable
(Rustlers, Copper, EAQ, PH_LT) — mixed, no discernible pattern. At
Gothic_ME specifically, the apparent cost is not distinguishable from
the noise floor already measured at this gauge**, and should not be
reported as a `split_C_in`-specific regression without that caveat.

## 12. The recharge-fraction ceiling constraint: implemented and tested (Section 8 item, resolved)

Section 8 listed "a structural minimum-recharge-fraction constraint" as
identified but never implemented. Implementing it directly surfaced a
naming problem worth stating before the result: **Section 8's own
description doesn't match what Section 3's forced-sweep actually found.**
The model's freely-learned recharge fraction sits at 0.65-0.9 (Section
3) — already well above any literal *minimum* near 0.3. A true minimum
(`rf >= 0.3`) would be a no-op, since the model already clears it. What
Section 3 actually showed moves baseflow fraction toward the Hubbard
benchmark is a **ceiling** on recharge fraction (`rf <= ~0.3`), i.e. a
*floor* on the fast-reservoir fraction `f_f` (forcing *more* infiltration
toward the fast reservoir, not less). This is what was implemented —
`train.py`'s new `recharge_ceiling_penalty` — with the naming correction
documented directly in its docstring, not silently substituted.

**Implementation**: a soft penalty, same pattern as the existing storage
barrier — `ReLU((1-rf_ceiling) - f_f)^2`, calibrated the same way (raw
penalty on `results_meltlag_single` = 0.2046; weight=100 puts the
weighted penalty at ~4% of the ~500-scale data loss, the same target
range used for the smoothness and storage-barrier terms). Default
weight 0.0 (off) — every existing checkpoint's reproducibility is
unaffected; the term is a literal `+ 0.0 * penalty` no-op unless
explicitly requested. Retrained both variants (`results_rfceiling_single/split`)
with `--rf-weight 100.0 --rf-ceiling 0.3`, otherwise identical to the
current recommended checkpoint's exact training configuration
(`cl_weight=0.3`, `barrier_weight=60`, `barrier_threshold=1.0`,
`smooth_weight=150`, 1000 epochs, seed=0) — an isolated before/after test
of this one change
(`figures_final/investigation_trail/19_recharge_ceiling_before_after.png`).

**Does baseflow fraction move inside the benchmark band? No, not at this
weight.** Recharge fraction moved substantially in the intended
direction — mean 0.68→0.40 (single_C_in), 0.64→0.38 (split_C_in) — and
baseflow fraction moved down at every gauge in both variants (e.g.
single_C_in: PH_LT 76.2%→65.0%, Gothic_ME 72.7%→67.1%, EAQ 81.7%→75.1%)
— but **remains outside the 12-33% annual band at every single gauge in
both variants**, nowhere close. The soft penalty at this weight pulls
recharge fraction about a third of the way from its free optimum (~0.7)
toward the 0.3 target, not all the way — consistent with Section 3's own
finding that the data loss itself has a strong, monotonic preference for
high recharge fraction across the whole tested range, which a moderate
soft penalty only partially overcomes.

**Is the trade as costly with a soft ceiling as with the hard-forced
value? No — dramatically less costly, but also dramatically less
effective.** Section 3's hard-forced `rf=0.3` cost **140.9% more** data
loss (`q_loss + cl_weight*cl_loss`) than the free optimum. The soft
ceiling at weight=100 costs **+1.5%** more data loss for single_C_in and
**-0.2%** (i.e. no measurable cost, marginally better) for split_C_in —
roughly two orders of magnitude cheaper. Per-gauge fit quality is mostly
preserved or mixed rather than uniformly degraded: Copper's `Q_NSE`
*improves* in both variants (single: -0.34→-0.09; split: -0.33→-0.28),
Gothic_ME's structural collapse is essentially unchanged (still order
-1400), while PH_LT's `Q_NSE` degrades modestly in both variants (0.53→0.43
single, 0.51→0.41 split) — the outlet gauge pays the clearest, if still
small, real cost. Chloride fit is mixed but mostly stable or slightly
improved (e.g. EAQ split_C_in: -19.85→-17.84).

**Net finding**: a soft recharge-fraction ceiling is cheap (in data-loss
terms) precisely because it doesn't do very much — it nudges recharge
fraction partway toward the target without fully committing to it, and
the resulting baseflow-fraction movement, while real and directionally
correct, falls well short of the benchmark band at every gauge. Reaching
the benchmark band would require either a much higher penalty weight
(re-approaching the hard-forced case's cost, per Section 3's loss-surface
measurement) or a hard constraint instead of a soft one — this round
tested the soft, cheap end of that trade-off directly rather than
assuming its outcome, and found it insufficient on its own to resolve the
baseflow-fraction gap, though it is a genuine, low-cost partial
improvement that could be combined with, or dialed up beyond, this
round's weight in future work.
