# Tracer-Informed Physics-Informed Neural Networks for Partitioning Fast and Slow Flow Pathways in a Snow-Dominated Mountain Watershed

Physics-informed neural network (PINN) code and results for partitioning
streamflow into fast and slow flowpaths at five nested sub-basins of the
East River watershed (Colorado), trained jointly on observed discharge
**and** stream chloride — using chloride as a near-conservative tracer to
give the model an independent signal about flowpath partitioning that
discharge alone can't provide.

Several model families are implemented and compared on a genuinely
held-out split: a tracer-informed PINN (two variants, `single_C_in` and
`split_C_in`), a hydrology-only PINN ablation (no chloride loss), and
four non-physics baselines (LSTM, GRU, TCN, LightGBM). Two validation
regimes are included: a random 80/20 held-out split, and a
leave-one-block-out (chronological) blocked cross-validation.

## Summary

The PINN enforces mass-conserving routing through an explicit fast-
reservoir / slow-reservoir ODE system per sub-basin, with a learned,
forcing-dependent fast-reservoir recession rate. Discharge and chloride
are fit jointly (tracer-informed variant) or discharge alone (hydrology-
only ablation), and compared against an LSTM with no physical structure.
Under a random day-level train/test split the LSTM outperforms both PINN
variants; under leave-one-block-out (chronological, season-scale)
cross-validation this reverses at two of the five gauges — see
`FINDINGS.md` and `results_blocked_masked/` for the full analysis.

One gauge, Gothic_ME (East River at the Middle East confluence), has a
discharge record flagged as untrustworthy before 2018-2019 (see
`FINDINGS.md`). **The primary analysis masks Gothic_ME's discharge out of
every model's training loss** (`--mask-gauge-discharge Gothic_ME`,
implemented in `pinn/train_holdout.py`, `pinn/train_lstm_baseline.py`,
`pinn/train_blocked.py`, `pinn/train_lstm_blocked.py`); the original,
unmasked training run is retained throughout as a **sensitivity
analysis**, not the headline result. Figures and result directories
follow this convention: an unsuffixed name (e.g.
`results_comparative_tracer_single/`) is the unmasked sensitivity run; a
`_masked`-suffixed name or a path under `results_gothic_masked/` /
`results_blocked_masked/masked/` is the primary, Gothic_ME-masked run.
Gothic_ME's own discharge metrics, wherever reported, are not ordinary
held-out skill numbers and are always labeled as such.

## Repository structure

```
pinn/                           Source code
  data.py                       Data loading, sub-basin areas, routing topology
  model.py                      HydroPINN: the fast/slow-reservoir ODE model
  split.py, blocked_split.py    Train/test split logic (random 80/20; leave-one-block-out)
  train.py                      Original in-sample training (no held-out split)
  train_holdout.py              Random-split training (PINN), --mask-gauge-discharge support
  train_lstm_baseline.py        LSTM baseline training, random split
  train_gru_baseline.py         GRU baseline (same design as the LSTM, nn.GRU instead)
  train_tcn_baseline.py         TCN baseline (causal dilated-conv stack, no recurrence)
  train_lightgbm_baseline.py    LightGBM baseline (10 independent per-gauge/variable
                                 tree models -- not a joint multi-output model, see its
                                 docstring for why)
  train_blocked.py              Blocked (leave-one-block-out) PINN training
  train_lstm_blocked.py         Blocked LSTM training
  train_gru_blocked.py          Blocked GRU training
  train_tcn_blocked.py          Blocked TCN training
  train_lightgbm_blocked.py     Blocked LightGBM training
  blocked_floor_baseline.py     No-model linear-interpolation floor baseline (blocked CV)
  evaluate.py, metrics.py       NSE / KGE / PBIAS and evaluation helpers
  style.py                      Shared matplotlib style for every figure
  make_plots_*.py               Figure-generating scripts (one per figure group); a
                                 "_masked" suffix = the Gothic_ME-masked primary-result variant
  diagnostics_*.py              Standalone diagnostic checks referenced in FINDINGS.md
  *.ipynb                       Exploratory / conceptual-diagram notebooks

figures_final/                  Every rendered manuscript figure, each PNG with a sibling
                                 .md caption. See FIGURE_MANIFEST.md for the figure map.

results_*/                      Trained-checkpoint directories (model.pt, metrics.json,
                                 history.json, predictions.npz) needed to reproduce the
                                 figures and tables below. Unsuffixed names = unmasked
                                 sensitivity runs; results_gothic_masked/ and
                                 results_blocked_masked/masked/ = primary (Gothic_ME-
                                 masked) runs. results_blocked_masked/unmasked/ holds the
                                 blocked-CV sensitivity run.

results_ml_baselines/           GRU / TCN / LightGBM results. unmasked/ and masked/
                                 (random 80/20 split, 5 seeds each) mirror the
                                 unmasked-sensitivity / masked-primary convention above;
                                 blocked_masked/ holds the leave-one-block-out CV runs
                                 (4 folds x 3 seeds, masked only -- these three model
                                 types have no separate unmasked blocked baseline).
                                 ml_baselines_final_report.txt and
                                 ml_baselines_comparison.txt are the pooled numeric
                                 report; pool_ml_baselines.py reproduces it from the
                                 checkpoints. **Every GRU/TCN/LightGBM number cited in
                                 the manuscript comes from this directory** -- these
                                 three architectures have no results anywhere else in
                                 this repository.

Full_PINN_Code_figures_final_.ipynb   Chronological, commit-by-commit research log
paper1_manuscript_figures.ipynb       Self-contained notebook reproducing the 10
                                       manuscript figures (4-13) from the checkpoints above
FINDINGS.md                           Full research narrative, in sections
MODEL_ARCHITECTURES.md                Exact architectures / hyperparameters for every model
FIGURE_MANIFEST.md                    Every figure, captioned, mapped to its FINDINGS.md section
README_paper1_manuscript_figures.md   Notes on the manuscript-figures notebook
```

**Not included** (see "Data" below): the raw input data, and any output
derived from the gap-filled-chloride experimental variant
(`results_filledchloride_single/`, `uploaded_data_filledchloride/` in the
original working tree) — that variant was exploratory and is not part of
the primary or sensitivity analysis reported here.

## Setup

Python 3.11 (tested; 3.10+ should work). Install dependencies:

```bash
pip install -r requirements.txt
```

## Data

This repository does **not** include the raw input data (sub-basin
discharge, chloride, ET, isotopes, SNOTEL snow records, CLM land-model
output, sub-basin delineation). Most are sourced from ESS-DIVE and
require a free ORCID login to download:

| Dataset | Reference | DOI |
|---|---|---|
| Discharge, water years 2015–present | Carroll & Williams (2018) | [10.21952/WTR/1495380](https://doi.org/10.21952/WTR/1495380) |
| Discharge, water years 2019–2024 | Carroll et al. (2024) | [10.15485/1779721](https://doi.org/10.15485/1779721) |
| Chloride / anion data | Williams et al. (2024) | [10.15485/1668054](https://doi.org/10.15485/1668054) |
| Stable water isotopes | Williams et al. (2023) | [10.15485/1668053](https://doi.org/10.15485/1668053) |
| Sub-basin delineation | Bill et al. (2018) | [10.21952/WTR/1508403](https://doi.org/10.21952/WTR/1508403) |
| Evapotranspiration (CLM simulation output) | Tran et al. (2020) | [10.15485/1727258](https://doi.org/10.15485/1727258) |
| LiDAR collection, August 2015 | Wainwright & Williams (2017) | [10.21952/WTR/1412542](https://doi.org/10.21952/WTR/1412542) |
| SNOTEL Butte, site 380 (precip. accum., SWE/snow depth) | USDA NRCS | [NRCS Report Generator](https://wcc.sc.egov.usda.gov/reportGenerator/) — no DOI |

ESS-DIVE restricts every public dataset to one of two licenses, chosen
by the contributor at submission: **CC BY 4.0** (attribution required) or
**CC0 1.0** (public domain). Check each dataset's own landing page for
which applies before reuse; this project's own code here is MIT-licensed
regardless (see `LICENSE`), independent of the input data's terms.

After obtaining the data, place the files in a directory named
`uploaded_data/` at the repository root, matching the filenames
`pinn/data.py`'s `load_dataset()` expects (discharge, chloride, ET,
isotope, and SNOTEL/CLM timeseries, plus `subbasins_attributes.csv`).

## Reproducing the results

All training scripts are deterministic given `--seed`; all checkpoints
needed to regenerate every figure below are already included, so
**retraining is optional** — the plotting scripts will run directly
against the included `results_*/` checkpoints once `uploaded_data/` is in
place (forcing data, i.e. precipitation/ET/day-of-year, is needed even to
re-run inference on an existing checkpoint).

1. **(Optional) Retrain.** From `pinn/`:
   - In-sample recommended checkpoint: `python train.py` (writes `results_meltlag_single/` — note: reproducing this exact checkpoint also requires the project's specific `--melt-lag-days`, phase-partitioning and recharge-ceiling flags; see `MODEL_ARCHITECTURES.md` for the full, verified hyperparameter table per checkpoint).
   - Random-split comparative family (sensitivity run): `python train_holdout.py --cl-weight 1.0 --out ../results_comparative_tracer_single`; `python train_lstm_baseline.py --out ../results_comparative_lstm`.
   - Primary (Gothic_ME-masked) comparative family: add `--mask-gauge-discharge Gothic_ME` to the same commands, writing to `../results_gothic_masked/...`.
   - Blocked CV: `python train_blocked.py --fold {1,2,3,4} --cl-weight 1.0 [--mask-gauge-discharge Gothic_ME] --out ...`; `python train_lstm_blocked.py --fold {1,2,3,4} [--mask-gauge-discharge Gothic_ME] --out ...`; floor baseline needs no training: `python blocked_floor_baseline.py`.
   - GRU/TCN/LightGBM baselines (random split): `python train_gru_baseline.py [--mask-gauge-discharge Gothic_ME] --out ../results_ml_baselines/{unmasked,masked}/gru_seed0`; same pattern for `train_tcn_baseline.py` and `train_lightgbm_baseline.py`. Blocked versions: `python train_gru_blocked.py --fold {1,2,3,4} --mask-gauge-discharge Gothic_ME --out ../results_ml_baselines/blocked_masked/gru_fold1_seed0`; same pattern for `train_tcn_blocked.py` and `train_lightgbm_blocked.py`.
2. **Generate figures**, from `pinn/`, each writing to `../figures_final/`:
   - `python make_plots_final.py`, `python make_plots.py` — core results (Figures 4-9)
   - `python make_plots_comparative.py` / `make_plots_comparative_masked.py` — Figures 10, 11 (sensitivity / primary)
   - `python make_plots_multiseed.py` / `make_plots_multiseed_masked.py` — Figure 12
   - `python make_plots_splitcin_holdout.py` / `make_plots_splitcin_holdout_masked.py` — Figure 13
   - `python make_plots_investigation_trail.py`, `make_plots_rfceiling.py` — investigation-trail figures
   - Or open `paper1_manuscript_figures.ipynb` and run top-to-bottom for Figures 4-13 in one pass.
3. **Tables / summary numbers**: `results_gothic_masked/tables56_masked.py` (Tables 5/6, masked), `results_blocked_masked/pool_step1.py` and `pool_step2.py` (blocked-CV pooling), `results_blocked_masked/blocked_masked_summary.json` (machine-readable summary of every blocked-CV metric), `results_ml_baselines/pool_ml_baselines.py` and `ml_baselines_final_report.txt` (GRU/TCN/LightGBM numbers — see note below).

## Outputs → figures / tables

| Output | Figure / table |
|---|---|
| `figures_final/core/*` | Manuscript Figures 4-9 |
| `figures_final/comparative_analysis/01_nse_comparison_holdout*.png` | Figure 10 (unsuffixed = sensitivity, `_masked` = primary) |
| `figures_final/comparative_analysis/05_lstm_capacity_ablation*.png` | Figure 11 |
| `figures_final/comparative_analysis/06_multiseed_robustness*.png` | Figure 12 |
| `figures_final/comparative_analysis/07_split_cin_comparison*.png` | Figure 13 |
| `figures_final/investigation_trail/*` | Investigation-trail figures, see `FIGURE_MANIFEST.md` |
| `results_gothic_masked/tables5_6_masked_manuscript_format.txt` | Tables 5, 6 (primary, masked) |
| `results_blocked_masked/blocked_masked_final_report.txt`, `blocked_masked_summary.json` | Blocked-CV results (Table 3 and related numbers) |
| `results_ml_baselines/ml_baselines_final_report.txt`, `ml_baselines_comparison.txt` | **Every GRU/TCN/LightGBM number in the manuscript** (random-split and blocked-CV, masked and unmasked) |
| `FIGURE_MANIFEST.md` | Full figure-to-finding map, all 27 figures |

## Citation

If you use this code or these results, please cite:

```
[ZENODO DOI]
```

## Contact

Ayodeji Felix Isarinade — [Felix_Isarinade1@baylor.edu]
