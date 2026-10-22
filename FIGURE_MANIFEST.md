# Figure Manifest

Every figure in `figures_final/`, in a manuscript-appropriate reading
order (methods -> core results -> investigation-trail/mechanism figures
-> comparative analysis last), not alphabetical or generation order.
Each entry: filename, one-line caption summary (the bolded lead
sentence of its sibling `.md` caption -- see that file for the full
caption), and the `FINDINGS.md` section it supports.

**Total: 27 figures** (4 methods, 9 core results, 7 investigation-trail, 7 comparative).

## Part 1 — Methods (model architecture, training, inputs)

| # | Figure | Caption summary | FINDINGS.md section |
|---|---|---|---|
| 1 | `core/12_forcing_overview.png` | Forcing data overview. | Section 4 (phase-partitioning) & Section 6 (melt-timing-lag correction) |
| 2 | `core/01_loss_curves.png` | Training loss convergence, log scale. | Section 9 (recommended checkpoint) |
| 3 | `core/10_ode_exactness_check.png` | ODE exactness check. | Architecture justification (model.py's closed-form ODE solver; not tied to a specific numbered finding) |
| 4 | `core/02_storage_dynamics.png` | Storage dynamics, log scale. | Section 1 (original symptoms) & Section 2 (storage barrier) |

## Part 2 — Core Results (fit quality, learned parameters, physical validation)

| # | Figure | Caption summary | FINDINGS.md section |
|---|---|---|---|
| 5 | `core/03_flow_partitioning.png` | Fast/slow flow partitioning, local generation. | Section 1 (the core fast/slow partitioning question) |
| 6 | `core/04_learned_parameters.png` | Learned physical parameters. | Section 3 (K_f/K_s recession-rate finding) |
| 7 | `core/05a_scatter_obs_vs_sim_single_Cin.png` | Goodness-of-fit scatter, single-C_in model. | Section 9 (recommended checkpoint, single-C_in) |
| 8 | `core/05b_scatter_obs_vs_sim_split_Cin.png` | Goodness-of-fit scatter, split-C_in model. | Section 9 (recommended checkpoint, split-C_in) |
| 9 | `core/06_model_comparison_outlet.png` | Single- vs split-C_in comparison at the outlet. | Section 9 (recommended checkpoint, variant comparison) |
| 10 | `core/07_evapoconcentration.png` | Evapoconcentration check. | Section 1 / Section 5 (evapoconcentration mechanism) |
| 11 | `core/08_baseflow_fraction_validation.png` | Baseflow fraction validation. | Section 3 & Section 9 (baseflow-fraction benchmark, unresolved open item) |
| 12 | `core/09_storage_floor_diagnostic.png` | Storage-floor / routing-floor diagnostic. | Section 2 (storage barrier: floor-crash diagnostic) |
| 13 | `core/11_isotope_cross_validation.png` | Isotope cross-validation (exploratory). | Section 9 (isotope cross-validation, never confirmatory) |

## Part 3 — Investigation Trail (the diagnostic narrative behind each fix)

| # | Figure | Caption summary | FINDINGS.md section |
|---|---|---|---|
| 14 | `investigation_trail/15_barrier_sweep_heatmap.png` | Barrier weight x threshold sweep | Section 2 (storage barrier re-sweep) |
| 15 | `investigation_trail/13_recession_curve_analysis.png` | Master recession-curve analysis | Section 3 (recession-curve mismatch that motivated K_f(t)) |
| 16 | `investigation_trail/17_rustlers_dec2015_phase_partition_fix.png` | Rustlers Dec 2015, before/after the phase-partitioning fix | Section 4 (missing-snowmelt double-counting fix) |
| 17 | `investigation_trail/14_unified_low_storage_mechanism.png` | Unified low-storage evapoconcentration mechanism | Section 5 (Rustlers/Copper/EAQ chloride-spike unification) |
| 18 | `investigation_trail/16_phlt_peak_before_after_meltlag.png` | PH_LT peak, before/after the melt-timing-lag fix | Section 6 (PH_LT peak timing fix) |
| 19 | `investigation_trail/18_inference_vs_retrain_taxonomy.png` | Inference-time-vs-retrain taxonomy | Section 10 (shared-parameter/inference-vs-retrain finding) |
| 20 | `investigation_trail/19_recharge_ceiling_before_after.png` | Recharge-fraction ceiling constraint, before/after | Section 8/12 (recharge-fraction ceiling constraint, implemented and tested) |

## Part 4 — Comparative Analysis (tracer PINN vs. hydrology-only PINN vs. LSTM)

| # | Figure | Caption summary | FINDINGS.md section |
|---|---|---|---|
| 21 | `comparative_analysis/01_nse_comparison_holdout.png` | Headline 3-way comparison, held-out test data. | Section 11.2 (headline held-out result) |
| 22 | `comparative_analysis/05_lstm_capacity_ablation.png` | LSTM capacity ablation. | Section 11.3 (chloride overfitting: capacity ablation + discharge negative control) |
| 23 | `comparative_analysis/06_multiseed_robustness.png` | Multi-seed robustness on the headline comparison. | Section 11.7 (multi-seed robustness check) |
| 24 | `comparative_analysis/02_baseflow_fraction_comparison.png` | Baseflow fraction: tracer PINN vs. hydrology-only PINN. | Section 11.4 (structurally PINN-only diagnostics) |
| 25 | `comparative_analysis/03_isotope_crossval_comparison.png` | Isotope cross-validation, tracer PINN vs. hydrology-only PINN | Section 11.4 (structurally PINN-only diagnostics) |
| 26 | `comparative_analysis/07_split_cin_comparison.png` | Split-C_in variant added to the held-out comparison. | Section 11.8 (does split_C_in narrow the gap to the LSTM?) |
| 27 | `comparative_analysis/04_summary_table.png` | 3-way comparison summary. | Section 11.5 (bottom line, consolidated) |

