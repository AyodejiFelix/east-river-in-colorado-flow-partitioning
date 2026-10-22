# Handoff Summary

Orientation document for anyone picking up this project for manuscript
drafting. Short by design — for the full technical narrative see
`FINDINGS.md`, and for the figures see `FIGURE_MANIFEST.md`.

## 1. What is, and isn't, in this repo

This repo contains the corrected mathematical formulation, all data
processing code, all model code (the tracer-informed PINN, the
hydrology-only ablation, and the LSTM baseline), the full training and
diagnostic history, and `FINDINGS.md` (the consolidated research
narrative, 12 sections — Section 11 alone has 8 subsections covering the
held-out comparative analysis). **It does not contain the original research
question, the literature framing, the dataset acquisition documentation,
or the full citation list** — those were produced separately and are not
part of this repository. Anyone picking this repo up for manuscript
drafting will need to supply that framing material separately.

Do not assume this repo is self-contained — it is the implementation,
experimentation, and results half of the project, not the framing half.

## 2. What was actually found

The project built a tracer-informed physics-informed neural network (PINN)
for the East River watershed (Colorado) to partition streamflow into
fast and slow flowpaths at five nested sub-basins, using both discharge
and stream chloride as joint training targets — the premise being that
chloride, as a near-conservative tracer, gives the model an independent
signal about flowpath partitioning that discharge alone can't provide.
Ten-plus rounds of hypothesis-driven diagnostic work traced a fully
causal chain from three original symptoms (storage collapsing to a
numerical floor, unbounded chloride spikes, and a persistent baseflow-
fraction mismatch against literature) back to their actual mechanisms: a
single fixed fast-reservoir recession constant that couldn't span both
storm response and winter recession, a missing snow/rain phase-partition
that double-counted precipitation as both immediate runoff and later
snowmelt, a melt-timing lag traced to the single-station SNOTEL proxy,
and a low-storage evapoconcentration mechanism (concentration rising
arithmetically as storage empties while tracer mass stays passively
stable) that explained chloride spikes at three separate gauges as one
mechanism, not three. Each fix was tested directly — cheaply first via
inference-only forward passes where possible, then via full retrains —
and each one's actual before/after effect was measured and reported, not
assumed. The final round added a genuinely held-out comparison (a fixed
80/20 split, never trained on) against a hydrology-only ablation (same
architecture, no tracer loss) and an LSTM baseline (no physics at all):
the LSTM wins on raw held-out prediction accuracy at every gauge, for
both discharge and chloride — but a capacity ablation (a second LSTM at
1/10th the parameters) showed the chloride "win" is overfitting specific
to the four data-scarce chloride gauges (38–43 observations each), not a
general architecture advantage — a discharge negative control at the
same reduced capacity showed no equivalent degradation. The PINN, in
contrast, produces the only output in the comparison that can be checked
against anything external to its own training data: baseflow fraction
against the Hubbard et al. benchmark, and fast/slow partitioning against
independent isotope tracer data — checks the LSTM has no architecture to
even attempt. A follow-up round confirmed the LSTM's discharge win is
robust across 5 seeds at every gauge, but its chloride win is not robust
at Gothic_ME specifically (the LSTM's own seed-to-seed volatility there
is large enough that its worst seed loses to the tracer PINN's typical
result); tested the two-end-member (`split_C_in`) tracer variant under
the same held-out split and found no consistent advantage over the
simpler `single_C_in` version (narrows the gap to the LSTM at only 2 of
5 gauges per metric; the one apparent Gothic_ME regression turned out,
on follow-up, to fall within that gauge's already-documented
seed-to-seed noise floor, not a confirmed split_C_in-specific defect);
and implemented the previously-deferred recharge-fraction
constraint (Section 8/12) — correctly specified as a *ceiling* on
recharge fraction, not a minimum (Section 8's original framing didn't
match what Section 3's forced-sweep had actually found). It moves
baseflow fraction substantially in the right direction at low aggregate
data-loss cost, but does introduce a real, if small, discharge-accuracy
cost at PH_LT specifically, does not reach the benchmark band at any
gauge at the tested weight, and was tested and reported but not adopted
into the recommended checkpoint (default weight 0.0/off).

## 3. Known limitations, honestly stated

- **Baseflow fraction never reached the Hubbard et al. benchmark**
  (12–33% annual), despite being the most-investigated single question in
  the project and despite a fully-traced diagnostic chain — the
  recommended checkpoint sits at 72–83% annual at every gauge. The
  mechanism (recharge fraction) is understood; a soft structural ceiling
  constraint (Section 12) moves it substantially in the right direction
  at both gauges tested, cheaply, but does not close the gap at the
  weight tested — a stronger constraint remains the indicated next step.
- **PH_LT's peak magnitude/shape gap** (2015's snowmelt peak reaches only
  ~37–39% of observed magnitude, even after the timing fix) is attributed
  to single-station SNOTEL spatial-representativeness limits — a data
  limitation, not an unresolved bug, and not expected to improve without
  a per-sub-basin SWE product.
- **Gothic_ME's discharge NSE is structurally, pathologically negative**
  (order -1300 to -2300) — a known artifact of NSE at a large-area,
  near-zero-flow confluence gauge, reproduced identically across multiple
  independently-trained checkpoints, unrelated to any fix tried.
- **Isotope (d-excess) cross-validation has never shown a strengthened
  correlation in any round** of this project — correlations stay weak
  (|r| typically <0.3) and inconsistent in sign across gauges. Not
  disconfirming the model's fast/slow partitioning, but never
  confirmatory either.

## 4. Source material for drafting

This file, `FIGURE_MANIFEST.md`, and `MODEL_ARCHITECTURES.md` (exact
architectures, parameter counts, and training hyperparameters for all
three models, verified directly against code and checkpoints — the
source for a manuscript Methods section) are the complete source
material for manuscript drafting from this repo's side of the project.
