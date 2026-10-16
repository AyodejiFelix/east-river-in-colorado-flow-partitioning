# Neural Network Architectures and Hyperparameter Settings

Precise specification of all three models used in this study — the
tracer-informed PINN, the hydrology-only PINN ablation, and the LSTM
baseline (including its capacity-ablation variant) — plus the training
configuration each was actually run with. Every number below was
verified directly against `pinn/model.py`, `pinn/train.py`,
`pinn/train_holdout.py`, `pinn/train_lstm_baseline.py`, and the
`train_config` block of each checkpoint's own `metrics.json` — not
recalled from memory or copied from an earlier description.

---

## 1. Tracer-informed PINN and hydrology-only PINN (shared architecture)

The hydrology-only PINN is **architecturally identical** to the
tracer-informed PINN — same class (`HydroPINN`), same parameter count,
same forward pass. The only difference is the training loss: chloride's
weight (`cl_weight`) is set to 0 for the hydrology-only variant, so
`Cin_fast`/`Cin_gw` receive no gradient and the tracer output is
untrained (Section 11.1). Both are documented together here; the
distinction is training configuration, in Section 3 below.

### 1.1 Overall structure

Two linear reservoirs per sub-basin (fast `S_f`, slow `S_s`), 5 modeled
nodes (Rustlers, Copper, EAQ, Gothic_ME, PH_LT), combined via a routed
network topology (not a flat sum). State updates use the **exact
closed-form solution** of the forced linear ODE `dy/dt = const(t) - K*y`
at every daily step:

```
y(t+1) = y(t)*exp(-K) + (const/K)*(1 - exp(-K))
```

— not a numerical integrator (Euler/RK4) and not a soft collocation
residual. This is applied identically to both the water-storage state
(`S_f`, `S_s`) and the tracer-mass state (`M_f`, `M_s`), using the same
`K` for both, per the governing equations.

### 1.2 Learned components

**A. `PartitionNet`** — the only true neural network in the architecture.
Outputs the two partition fractions `f_uf,j(t)` (ultra-fast bypass) and
`f_f,j(t)` (fast-vs-slow infiltration split) as functions of forcing and
sub-basin identity.

| Layer | Type | Shape | Params |
|---|---|---|---|
| `embed` | `nn.Embedding(5, 4)` | 5×4 | 20 |
| `net.0` | `nn.Linear(8, 32)` | 32×8 + 32 | 256 + 32 = 288 |
| `net.1` | `nn.Tanh()` | — | 0 |
| `net.2` | `nn.Linear(32, 32)` | 32×32 + 32 | 1024 + 32 = 1056 |
| `net.3` | `nn.Tanh()` | — | 0 |
| `net.4` | `nn.Linear(32, 2)` | 2×32 + 2 | 64 + 2 = 66 |
| **Total** | | | **1430** |

- **Input** (8-dim, per sub-basin per day): 4-dim sub-basin embedding +
  `P_e,j(t)/mean(P_e)` + `ET_j(t)/mean(ET)` + `sin(2*pi*doy/366)` +
  `cos(2*pi*doy/366)`.
- **Output**: `f_uf,j(t) = sigmoid(out[0])`, `f_f,j(t) = sigmoid(out[1])`
  — both bounded in (0,1) by construction (guarantees mass conservation
  of the split with no separate penalty needed).
- **Activation**: `Tanh` in both hidden layers; `sigmoid` on the output
  heads.
- Evaluated once per day, per sub-basin, inside the sequential 737-step
  forward loop (shared weights across all sub-basins and all days —
  sub-basin identity enters only through the embedding).

**B. Learnable scalar parameters** — 9 per sub-basin (single_C_in) or 10
(split_C_in), all unconstrained (`raw_*`) and mapped to physically
bounded ranges via `sigmoid`/`softplus`:

| Parameter | Raw name | Transform | Bounded range | Meaning |
|---|---|---|---|---|
| `K_f,recession,j` | `raw_Kf_recession` | `_softplus_scale` (sigmoid-based) | 0.001 – 0.3 /day | Fast-reservoir recession rate, dry periods |
| `K_f,storm,j` | `raw_Kf_storm` | `_softplus_scale` | 0.02 – 1.5 /day | Fast-reservoir recession rate, storm response |
| `a_j` | `raw_Kf_a` | `softplus` | (0, inf), 1/mm | `K_f(t)` transition sharpness |
| `b_j` | `raw_Kf_b` | `softplus` | (0, inf), mm/day | `K_f(t)` transition threshold (on `P_e`) |
| `K_s,j` | `raw_Ks` | `_softplus_scale` | 0.001 – 0.3 /day | Slow-reservoir recession rate (constant, not time-varying) |
| `alpha_ET,j` | `raw_alphaET` | `sigmoid` | (0, 1) | Fraction of ET drawn from the fast reservoir |
| `C_in,fast,j` | `raw_Cin_fast` | `0.05 + softplus` (single) / `0.02 + softplus` (split) | (0.05, inf) or (0.02, inf) mg/L | Fast-pathway chloride end-member |
| `C_in,gw,j` | `raw_Cin_gw_delta` (split only) | `Cin_fast + 0.05 + softplus(delta)` | ≥ `Cin_fast` + 0.05 mg/L | Groundwater-recharge chloride end-member (split_C_in only; single_C_in sets `Cin_gw = Cin_fast`) |
| `S0_f,j` | `raw_S0f` | `1.0 + softplus` | (1.0, inf) mm | Initial fast-reservoir storage |
| `S0_s,j` | `raw_S0s` | `5.0 + softplus` | (5.0, inf) mm | Initial slow-reservoir storage |

`K_f,j(t)` itself is **not** a free function of time — it interpolates
between the two learned bounds via a fixed sigmoid form:

```
K_f,j(t) = K_f,recession,j + (K_f,storm,j - K_f,recession,j) * sigmoid(a_j * (P_e,j(t) - b_j))
```

`K_s,j` and `alpha_ET,j` are learned sub-basin **constants**, not
functions of time, per the corrected formulation's explicit requirement.

### 1.3 Total parameter count (verified by direct instantiation)

| Variant | `PartitionNet` | Scalar params (5 sub-basins × N) | Total |
|---|---|---|---|
| `single_C_in` (`split_cin=False`) | 1430 | 45 (9 × 5) | **1,475** |
| `split_C_in` (`split_cin=True`) | 1430 | 50 (10 × 5) | **1,480** |

Both are used by the tracer-informed PINN (two variants) and by the
hydrology-only PINN (same class, `cl_weight=0` — this study only used
`split_cin=False` for the hydrology-only ablation, so its parameter
count is **1,475**, identical to the tracer-informed `single_C_in`
model).

### 1.4 Recharge-fraction ceiling (optional regularizer, off by default)

Added this round (`pinn/train.py`'s `recharge_ceiling_penalty`), not part
of the architecture proper — a soft penalty term, same functional form as
the existing storage barrier:

```
recharge_ceiling_penalty(f_f, rf_ceiling) = ReLU((1 - rf_ceiling) - f_f)^2 . mean()
```

Default weight 0.0 (off, no effect on any checkpoint that doesn't
explicitly request it). Tested at `rf_weight=100.0, rf_ceiling=0.3`
(`results_rfceiling_single/split` — Section 12, FINDINGS.md); not part of
the recommended checkpoint.

---

## 2. LSTM baseline (`LSTMBaseline`, `pinn/train_lstm_baseline.py`)

A single joint LSTM over all 5 sub-basins' forcing at once — not 5
independent per-basin models, since the routed gauges (Gothic_ME, PH_LT)
measure upstream contributions that basin-local forcing alone can't
explain.

### 2.1 Architecture

| Layer | Type | Shape | Params (hidden=64) |
|---|---|---|---|
| `lstm` | `nn.LSTM(input_size=12, hidden_size=H, num_layers=1, batch_first=True)` | 4 gates × H × (12+H) + 4 gates × H (biases, ×2 for `ih`/`hh`) | 19,712 (`weight_ih_l0`: 3072, `weight_hh_l0`: 16384, `bias_ih_l0`: 256, `bias_hh_l0`: 256) |
| `head` | `nn.Linear(H, 10)` | 10×H + 10 | 650 (H=64: 640+10) |
| Output split | `softplus(head_out)`, sliced `[:5]` = Q, `[5:]` = Cl | — | 0 |

- **Input** (12-dim, per day, all 5 sub-basins jointly): `P_e,1..5(t)`
  (z-scored) + `ET_1..5(t)` (z-scored) + `sin(2*pi*doy/366)` +
  `cos(2*pi*doy/366)`. Sequence length 737 (full record), batch size 1
  (single continuous sequence).
- **Output**: `softplus` applied to the 10-dim head output, split into 5
  discharge predictions and 5 chloride predictions (one per sub-basin,
  per day) — non-negativity is the only physical constraint; no other
  structure (no reservoirs, no routing, no partition network).
- **Activation**: standard LSTM gating (sigmoid/tanh internal to
  `nn.LSTM`) + `softplus` on the final output only.

### 2.2 Total parameter count (verified by direct instantiation)

| Variant | `hidden` | `lstm` params | `head` params | Total |
|---|---|---|---|---|
| Main baseline | 64 | 19,968 | 650 | **20,618** |
| Capacity ablation | 16 | 1,920 (`weight_ih_l0`:768, `weight_hh_l0`:1024, `bias_ih_l0`:64, `bias_hh_l0`:64) | 170 | **2,090** |

The capacity-ablation LSTM has ~10x fewer parameters than the main
LSTM baseline (2,090 vs. 20,618) — but for a size comparison against the
PINN, note both LSTM variants have *more* parameters than either PINN
variant (1,475/1,480): even the 10x-smaller ablation LSTM (2,090) exceeds
the PINN's parameter count. The "PINN has ~2 orders of magnitude fewer
parameters" comparisons elsewhere in this project's figures/FINDINGS.md
refer specifically to the PINN vs. the hidden=64 LSTM (1,475–1,480 vs.
20,618), not vs. the capacity-ablation variant.

### 2.3 Known dead CLI flag

`train_lstm_baseline.py` exposes `--dropout` (default 0.0) but
`LSTMBaseline.__init__` does not accept or use a `dropout` argument at
all — the flag is parsed but has no effect on the model. Since it
defaults to 0.0 and was never set to anything else in this study, this
had no effect on any reported result, but is documented here for
methodological completeness rather than silently omitted.

---

## 3. Training configuration by checkpoint family

All three model classes are trained with **Adam** and a
**`CosineAnnealingLR`** schedule (`T_max = epochs`, decaying to 0 at the
final epoch), with gradient-norm clipping at 5.0 for the PINN variants
(`torch.nn.utils.clip_grad_norm_`, max norm 5.0; the LSTM script applies
the same clip). Loss is a per-gauge-averaged, std-normalized MSE
(`masked_norm_mse`), computed separately for discharge and chloride and
combined with `cl_weight`, so a densely-observed gauge (e.g. PH_LT, 610
Q obs) cannot dominate the gradient over a sparse one (e.g. Rustlers, 74
Q obs) — every one of the 5 nodes gets equal weight regardless of
observation count. The first 30 days (`SPINUP_DAYS`) are excluded from
the loss (state warm-up) but still integrated.

Exact configurations differ by checkpoint family — verified directly
from each checkpoint's own `metrics.json.train_config`, not assumed
uniform:

| Checkpoint family | Script | Variant(s) | Epochs | LR | `cl_weight` | `smooth_weight` | `barrier_weight` / `threshold` | `rf_weight` / `ceiling` | Seed(s) | Held-out split | Grad clip |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `results_meltlag_single/split` (recommended checkpoint) | `train.py` | single, split | 1000 | 3e-3 | **0.3** | 150.0 | 60.0 / 1.0mm | 0 (off) | 0 | No (100% of data) | 5.0 |
| `results_rfceiling_single/split` (tested, not adopted) | `train.py` | single, split | 1000 | 3e-3 | 0.3 | 150.0 | 60.0 / 1.0mm | **100.0 / 0.3** | 0 | No | 5.0 |
| `results_comparative_tracer_single/split` | `train_holdout.py` | single, split | 1000 | 3e-3 | **1.0** | 150.0 (default) | 60.0 / 1.0mm (default) | 0 (off) | 0 (+1, 2 for single) | **Yes** (seed=42, 80/20) | 5.0 |
| `results_comparative_hydro_only` | `train_holdout.py` | single | 1000 | 3e-3 | **0.0** | 150.0 (default) | 60.0 / 1.0mm (default) | 0 (off) | 0, 1, 2 | Yes (seed=42, 80/20) | 5.0 |
| `results_comparative_lstm` (+ `_seed{1,2,3,4}`) | `train_lstm_baseline.py` | hidden=64 | **3000** | **1e-3** | n/a (joint Q+Cl loss, unweighted sum) | n/a | n/a | n/a | 0, 1, 2, 3, 4 | Yes (seed=42, 80/20) | 5.0 |
| `results_comparative_lstm_small` | `train_lstm_baseline.py` | hidden=16 | 3000 | 1e-3 | n/a | n/a | n/a | n/a | 0 | Yes (seed=42, 80/20) | 5.0 |

Additional LSTM-specific hyperparameter: `weight_decay=1e-4` (Adam), all
LSTM runs. `dropout=0.0` (parsed but inert, Section 2.3).

**Note on `cl_weight`, since it varies by checkpoint family and matters
for interpreting any cross-checkpoint comparison**: the recommended
checkpoint (`results_meltlag_*`) and its recharge-ceiling variant use
`cl_weight=0.3`; every checkpoint in the 3-/4-way comparative analysis
(Section 11, `results_comparative_*`) uses `cl_weight=1.0` for the
tracer-informed variant and `0.0` for hydrology-only. This is a
deliberate difference between the two families, not an inconsistency:
the comparative-analysis round used `train_holdout.py`'s own default
(1.0) for a full-weight chloride signal under the held-out split, while
the recommended checkpoint's `0.3` reflects the separate calibration
history documented in `pinn/README.md`. Any manuscript text quoting a
`cl_weight` value should specify which checkpoint family it refers to.

### 3.1 Held-out split (comparative-analysis checkpoints only)

Fixed-seed (42), stratified random 80/20 split of each gauge's own
observed days, independently for Q and Cl, built once
(`pinn/split.py`, `results_comparative/holdout_split.npz`) and loaded
identically by every checkpoint in this family — not regenerated per
run. Per-gauge normalization statistics (`q_std`, `cl_std`) used inside
the loss are computed from the train-split observations only.

### 3.2 Compute cost (measured, not estimated)

| Model | Params | Epochs | Wall time (this environment's CPUs) |
|---|---|---|---|
| PINN (either variant) | 1,475 – 1,480 | 1000 | ~2000 – 2040s (~34 min) |
| LSTM (hidden=64) | 20,618 | 3000 | 55.8s |
| LSTM (hidden=16) | 2,090 | 3000 | 36.9s |

The PINN's cost is dominated by its sequential 737-step Python-level
forward loop (the exact closed-form ODE update, once per day); the LSTM
uses a single vectorized recurrent-kernel call per forward pass.
