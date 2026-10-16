# Paper 1 — Manuscript Figures Notebook

`paper1_manuscript_figures.ipynb` is a single, fully-executed Jupyter
notebook that reproduces the manuscript's figures directly from this
codebase's plotting scripts, with outputs saved and embedded in the
notebook itself. GitHub renders a `.ipynb` file's saved outputs inline —
click the file in this repo and the figures display immediately, with
nothing to run.

## Scope

This notebook covers the **10 manuscript figures produced by
Python/matplotlib code in this repository**: **Figures 4 through 13.**

Three figures are intentionally excluded — not overlooked:

- **Figure 1** (sub-basin delineation map) is a **GIS-generated map**
  (ArcGIS/QGIS, projected UTM Zone 13N, drawn from the watershed's GIS
  attribute table), not a Python/matplotlib output, so it falls outside
  this notebook's scope for the same reason Figures 2–3 do.
- **Figures 2 and 3** are the two conceptual diagrams (water/tracer-balance
  schematic; model-architecture diagram) — built separately, outside this
  codebase's plotting scripts.

No retraining happens anywhere in this notebook. Every figure is produced
by loading an existing trained checkpoint and running inference only, or
(for Figures 10–13) by reading each checkpoint's already-computed
`metrics.json` directly.

## Figure map

| Figure | What it shows | Source script |
|---|---|---|
| 4 | Locally generated flow by pathway (Q_uf/Q_f/Q_s) and the resulting fast-flow fraction, all 5 sub-basins | `make_plots_final.py` |
| 5 | Learned fast- and slow-reservoir storage (S_f, S_s), log scale | `make_plots_final.py` |
| 6 | Unified low-storage evapoconcentration mechanism (S_s, tracer mass M_s, concentration C_s) at Rustlers/Copper/EAQ | `make_plots_investigation_trail.py` |
| 7 | PH_LT outlet peak, before vs. after the melt-timing-lag correction (2014/2015 snowmelt) | `make_plots_investigation_trail.py` |
| 8 | Routed baseflow fraction vs. the Carroll et al. (2018) benchmark band | `make_plots_final.py` |
| 9 | Baseflow fraction before/after the recharge-fraction ceiling constraint (single- and split-C_in) | `make_plots_rfceiling.py` |
| 10 | Held-out discharge/chloride NSE: tracer-informed PINN vs. hydrology-only PINN vs. LSTM baseline | `make_plots_comparative.py` |
| 11 | LSTM capacity ablation (hidden=64 vs. hidden=16) — is the chloride collapse about model size or data scarcity? | `make_plots_comparative.py` |
| 12 | Multi-seed robustness of the 3-way held-out comparison (NSE, mean ± seed range) | `make_plots_multiseed.py` |
| 13 | 4-way held-out comparison adding the split-C_in tracer PINN variant | `make_plots_splitcin_holdout.py` |

## Checkpoints used

| Checkpoint | Used by |
|---|---|
| `results_meltlag_single`, `results_meltlag_split` | Figures 4, 5, 8, 9 (before), 7 (after) |
| `results_phasepartition_single` | Figure 7 (before) |
| `results_rfceiling_single`, `results_rfceiling_split` | Figure 9 (after) |
| `results_comparative_tracer_single` | Figures 10, 11, 12, 13 |
| `results_comparative_hydro_only` | Figures 10, 12 |
| `results_comparative_lstm`, `results_comparative_lstm_small` | Figures 10, 11, 12 |
| `results_comparative_lstm_seed1`–`seed4`, `results_comparative_tracer_single_seed1`–`seed2`, `results_comparative_hydro_only_seed1`–`seed2` | Figure 12 (multi-seed) |
| `results_comparative_tracer_split` | Figure 13 |

All checkpoints use the identical, fixed 80/20 held-out split (`split.py`,
seed=42) where applicable — no model in the comparison ever trains on the
days its held-out metrics are computed from.

## Notes carried over from the manuscript

- **Figure 8 / Figure 9**: the 12–33% (mean annual) / 17–50%
  (baseflow-period, Dec–Mar) benchmark band is from **Carroll et al.
  (2018)** — corrected from an earlier round's misattribution to "Hubbard
  et al. (2018)."
- **Gothic_ME's extreme discharge NSE/KGE values** (order −2,000 to
  −11,000, Figures 10 and 12): a known, documented structural issue —
  small/near-zero observed discharge at that gauge — not a bug. See
  manuscript Section 3.7.
- **Figure 6's use of the median, not the mean**: `C_s` has no ceiling in
  the current code (unlike `C_f`) and spikes to millions of mg/L on a
  handful of days when `S_s` hits its numerical floor; the median
  represents the other 99%+ of the record.
- **Figure 12's PINN seed count (3) vs. LSTM's (5)**: a disclosed,
  compute-driven scope reduction — each additional PINN seed costs ~34 min
  on this environment's CPUs vs. ~1 min for an LSTM seed.

## Corrections applied in this revision

An earlier draft of this notebook proposed a Figure 1/4–13 mapping before
the current manuscript draft had been checked directly. Verifying it
against the manuscript's actual captions found it wrong for 10 of 11
slots (only the storage-dynamics figure coincidentally matched). This
version replaces every mismatched figure/caption and drops an earlier,
invented "multi-seed KGE/PBIAS" figure entirely — no such figure exists in
the manuscript (KGE is reported in Table 5, baseflow PBIAS in Table 2).

Two figures were checked against the manuscript's full text before being
confirmed as correctly excluded: `09_storage_floor_diagnostic.png` and
`10_ode_exactness_check.png` are genuinely diagnostic-only and do not
correspond to any manuscript figure under any name.

## Running it yourself

The notebook is self-contained given the repo layout: it imports directly
from `data.py`, `model.py`, and `style.py` (same modules the `.py`
plotting scripts use, so there's no risk of drift), and expects
`uploaded_data/` and the `results_*` checkpoint directories as siblings of
`pinn/`. Open it with `pinn/` as the working directory and run top to
bottom — no training required, every figure loads an existing checkpoint.

## Verification

Executed end-to-end via `nbconvert --execute --inplace`: zero errors, all
10 figure cells produced valid embedded PNG outputs (magic bytes
confirmed), and spot-checked figures were visually compared against the
manuscript's own reported numbers (e.g., Figure 8's per-gauge baseflow
percentages match the manuscript caption's "72–83% annual, 68–84% winter"
range exactly). Also verified standalone from a clean, isolated
extraction of the notebook + its required files, independent of the live
development environment.
