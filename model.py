"""
Tracer-informed Physics-Informed Neural Network for the East River
fast/slow flow partitioning problem, implementing the corrected governing
equations (Sections 4-5 of 02_corrected_mathematical_formulation.docx).

Design
------
The governing equations are two linear reservoirs per sub-basin driven by
a partitioning of effective water input that is NOT directly observable
(how much of P_e goes to ultra-fast surface runoff vs. infiltration, and
how infiltration splits between the fast and slow reservoirs). That
partitioning is the learned part of the model: a small neural network,
conditioned on forcing and a per-sub-basin embedding, outputs the two
partition fractions f_uf,j(t) and f_f,j(t) as sigmoids (hence physically
bounded in [0,1] by construction, guaranteeing mass conservation of the
split -- no separate conservation penalty is needed).

Given those fractions, the water-balance and tracer-mass-balance ODEs are
*linear* in S and M with piecewise-daily-constant forcing:

    dS/dt = C(t) - K*S          =>  exact closed form over one day:
    S(t+dt) = S(t) e^{-K dt} + (C/K) (1 - e^{-K dt})

Because Q_x,j = K_x,j * S_x,j (Section 5.3a), the tracer ODEs reduce the
same way: dM/dt = (inflow mass flux) - K*M, using the *same* K as the
water storage. So both state updates use the exact exponential-decay
solution of a forced linear ODE rather than a numerical stepper (Euler/RK4)
or a soft collocation residual -- this hard-codes the physics into the
architecture exactly (no discretization error, unconditionally stable for
any K > 0) rather than penalizing ODE residuals in the loss, which is the
usual (harder to train) collocation-PINN approach.

K_s,j and alpha_ET,j are learned as sub-basin CONSTANTS (not free
functions of time), per the corrected formulation's explicit requirement
in Section 1 ("Recommend constants per sub-basin ... if unconstrained, the
network can fit a different K at every timestep, which defeats the
purpose of embedding this as a physical constraint").

K_f,j(t) is the one exception, and it is NOT a free function of time --
a master-recession-curve analysis against observed discharge (see
README) showed a single constant K_f cannot span both storm-response and
winter-recession timescales at any of the 5 gauges (observed recession
constants sit 10-55x closer to K_s than to whatever single K_f the model
learned). Per Section 1's own caution against K becoming "an arbitrary
free function of time," K_f,j(t) is instead a small, bounded,
interpretable function of ONE named physical covariate -- effective
water input P_e,j(t), already a forcing term, not an arbitrary basis --
interpolating between two learned per-basin bounds:

    K_f,j(t) = K_f,recession,j + (K_f,storm,j - K_f,recession,j)
               * sigmoid(a_j * (P_e,j(t) - b_j))

so K_f,j(t) is fast just after a precipitation/melt event and relaxes
toward a slow, K_s-like recession rate during dry periods, with exactly
4 learned scalars per basin (K_f,recession,j, K_f,storm,j, a_j, b_j) --
still a small number of named, bounded constants, not a free function.

The five modeled nodes are then combined via the routed-network topology
of Section 4 (direct summation of volumetric discharge / tracer mass flux
across confluences, then C_tot = Phi_tot / Q_tot), not a flat sum across
all five sub-basins.
"""
import torch
import torch.nn as nn

from data import SUBBASINS, UPSTREAM, MM_PER_DAY_TO_CMS

N_SUB = len(SUBBASINS)
SUB_IDX = {b: i for i, b in enumerate(SUBBASINS)}

# See the derivation comment at the Ctot computation in forward() below.
CTOT_DENOM_FLOOR = 0.01  # cms
CTOT_CEILING_MGL = 1.8542 * 5  # observed max stream Cl x 5 safety margin, ~9.27 mg/L

# C_f (fast-reservoir concentration, Mf/Sf) has no barrier protecting Sf from
# crashing toward its floor the way Ss is protected, so Cf can spike without
# bound when Sf empties faster than Mf drains (found via Rustlers' Aug-Nov
# chloride investigation: Cf reached 2,044,598 mg/L in one checkpoint,
# confirmed NOT the driver of any observed metric since Qf collapses
# simultaneously and self-limits Cf's effect on Ctot -- but an unprotected
# intermediate value reaching 6-figure concentrations is a latent numerical-
# stability risk worth a defensive ceiling regardless). Same derivation
# pattern as CTOT_CEILING_MGL: the single highest chloride concentration
# ever recorded anywhere in this project's data (ER-PLM8 groundwater well,
# the anomalously saline deep-flowpath well flagged elsewhere in this file,
# 2617.49 umol/L = 92.7979 mg/L) x 5 safety margin.
CF_CEILING_MGL = 92.7979 * 5  # ~463.99 mg/L


def _softplus_scale(raw, lo, hi):
    """Map raw unconstrained param to (lo, hi) via sigmoid."""
    return lo + (hi - lo) * torch.sigmoid(raw)


class PartitionNet(nn.Module):
    """f_uf,j(t), f_f,j(t) as functions of forcing + sub-basin identity."""

    def __init__(self, n_sub=N_SUB, embed_dim=4, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(n_sub, embed_dim)
        in_dim = embed_dim + 4  # Pe_norm, ET_norm, sin(doy), cos(doy)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )

    def forward(self, sub_idx, pe_norm, et_norm, sin_doy, cos_doy):
        # all inputs broadcastable to shape [..., ] ; sub_idx: [N_SUB] long
        e = self.embed(sub_idx)  # [N_SUB, embed_dim]
        x = torch.cat(
            [e, pe_norm.unsqueeze(-1), et_norm.unsqueeze(-1),
             sin_doy.unsqueeze(-1), cos_doy.unsqueeze(-1)],
            dim=-1,
        )
        out = self.net(x)
        f_uf = torch.sigmoid(out[..., 0])
        f_f = torch.sigmoid(out[..., 1])
        return f_uf, f_f


class HydroPINN(nn.Module):
    def __init__(self, n_sub=N_SUB, split_cin=True):
        """split_cin=True: two learned chloride end-members, C_in,fast and
        C_in,gw (see README). split_cin=False: reproduces the original
        single-end-member model exactly (same init/formula as the first
        checkpoint in results/), for a fair apples-to-apples comparison
        between the two variants under an otherwise-identical architecture."""
        super().__init__()
        self.n_sub = n_sub
        self.split_cin = split_cin
        self.partition_net = PartitionNet(n_sub)

        # Learnable sub-basin constants (raw / unconstrained parameters).
        # K_f,j(t)'s two bounds (see class docstring): K_f,recession init
        # targets ~0.01/day, in the middle of the observed winter
        # recession range (0.004-0.014/day, see README's master-recession-
        # curve analysis); K_f,storm keeps the original single-K_f init
        # (bound 0.02-1.5/day, same as before) for storm response.
        # a/b are unconstrained reals mapped to positive via softplus:
        # a (1/mm, controls transition sharpness) inits near 5.0/mm --
        # sharp enough that Kf(0) sits close to Kf_recession rather than
        # blending far up toward Kf_storm even at zero input (verified
        # empirically before training: at a~1/mm, Kf(0) came out ~0.26/day,
        # nowhere near the 0.01/day target; a~5/mm gives Kf(0)~0.03/day);
        # b (mm/day, the P_e transition threshold) inits near 0.69 mm/day,
        # a modest measurable-precip/melt threshold.
        self.raw_Kf_recession = nn.Parameter(torch.zeros(n_sub) - 3.47)
        self.raw_Kf_storm = nn.Parameter(torch.zeros(n_sub))
        self.raw_Kf_a = nn.Parameter(torch.zeros(n_sub) + 5.0)
        self.raw_Kf_b = nn.Parameter(torch.zeros(n_sub))
        self.raw_Ks = nn.Parameter(torch.zeros(n_sub) - 1.0)
        self.raw_alphaET = nn.Parameter(torch.zeros(n_sub))
        # Two chloride end-members (doc Section 1, critique #5 / Section 7
        # open item): fast/preferential pathway water is close to fresh
        # event-water chemistry (low Cl), groundwater recharge has more
        # time for water-rock contact (higher Cl). Cin_gw is parameterized
        # as Cin_fast + a non-negative offset, hard-constraining
        # Cin_gw >= Cin_fast rather than hoping training discovers that
        # ordering. Initialized from the ER-PLM groundwater well chloride
        # record (see README): Cin_fast starts near the stream baseline
        # (~0.1 mg/L) and the offset starts near the robust well median
        # (~0.8 mg/L, excluding the two anomalously saline wells ER-PLM8/9
        # which look like a distinct deep flowpath, not general shallow
        # recharge) -- both remain learnable per sub-basin.
        # single_cin mode reproduces the ORIGINAL model's raw_Cin exactly
        # (init=0, formula 0.05+softplus below) so results/ is reproducible;
        # split_cin mode uses the well-informed init described above.
        if split_cin:
            self.raw_Cin_fast = nn.Parameter(torch.zeros(n_sub) - 2.25)
            self.raw_Cin_gw_delta = nn.Parameter(torch.zeros(n_sub))
        else:
            self.raw_Cin_fast = nn.Parameter(torch.zeros(n_sub))
        self.raw_S0f = nn.Parameter(torch.zeros(n_sub))
        self.raw_S0s = nn.Parameter(torch.zeros(n_sub) + 1.0)

        area_cms = torch.tensor([MM_PER_DAY_TO_CMS[b] for b in SUBBASINS], dtype=torch.float32)
        self.register_buffer("area_cms", area_cms)

        up_mask = torch.zeros(n_sub, n_sub)
        for j, b in enumerate(SUBBASINS):
            for u in UPSTREAM[b]:
                up_mask[j, SUB_IDX[u]] = 1.0
        self.register_buffer("up_mask", up_mask)  # up_mask[j, u] = 1 if u feeds j

    def physical_params(self):
        Kf_recession = _softplus_scale(self.raw_Kf_recession, 0.001, 0.3)  # 1/day, dry-period K_f
        Kf_storm = _softplus_scale(self.raw_Kf_storm, 0.02, 1.5)           # 1/day, storm-response K_f
        Kf_a = torch.nn.functional.softplus(self.raw_Kf_a)                 # 1/mm, transition sharpness
        Kf_b = torch.nn.functional.softplus(self.raw_Kf_b)                 # mm/day, P_e transition threshold
        Ks = _softplus_scale(self.raw_Ks, 0.001, 0.3)     # 1/day, slow reservoir recession
        alphaET = torch.sigmoid(self.raw_alphaET)          # fraction of ET from fast store
        if self.split_cin:
            Cin_fast = 0.02 + torch.nn.functional.softplus(self.raw_Cin_fast)  # mg/L, event water
            Cin_gw = Cin_fast + 0.05 + torch.nn.functional.softplus(self.raw_Cin_gw_delta)  # mg/L, recharge (>= Cin_fast)
        else:
            Cin_fast = 0.05 + torch.nn.functional.softplus(self.raw_Cin_fast)  # matches original single-Cin exactly
            Cin_gw = Cin_fast
        S0f = 1.0 + torch.nn.functional.softplus(self.raw_S0f)   # mm
        S0s = 5.0 + torch.nn.functional.softplus(self.raw_S0s)   # mm
        return Kf_recession, Kf_storm, Kf_a, Kf_b, Ks, alphaET, Cin_fast, Cin_gw, S0f, S0s

    @staticmethod
    def _Kf_of_t(Pe_t, Kf_recession, Kf_storm, Kf_a, Kf_b):
        """K_f,j(t) = K_f,recession,j + (K_f,storm,j - K_f,recession,j)
        * sigmoid(a_j * (P_e,j(t) - b_j)) -- see class docstring."""
        return Kf_recession + (Kf_storm - Kf_recession) * torch.sigmoid(Kf_a * (Pe_t - Kf_b))

    @staticmethod
    def _exp_step(S, K, const_input, dt=1.0):
        """Exact solution of dS/dt = const_input - K*S over one step."""
        decay = torch.exp(-K * dt)
        S_new = S * decay + (const_input / K) * (1.0 - decay)
        return S_new

    def forward(self, precip, et, doy, spinup_days=0):
        """
        precip, et: [T, n_sub] tensors (mm/day)
        doy: [T] tensor, day-of-year in [1, 366]
        Returns dict of per-node time series, each [T, n_sub] (local) or
        [T, n_sub] (routed totals, only meaningful for nodes with upstream).
        """
        T = precip.shape[0]
        device = precip.device
        Kf_recession, Kf_storm, Kf_a, Kf_b, Ks, alphaET, Cin_fast, Cin_gw, S0f, S0s = self.physical_params()

        pe_mean = precip.mean().clamp_min(1e-3)
        et_mean = et.mean().clamp_min(1e-3)
        sin_doy = torch.sin(2 * torch.pi * doy / 366.0)
        cos_doy = torch.cos(2 * torch.pi * doy / 366.0)

        sub_idx = torch.arange(self.n_sub, device=device)

        Sf = S0f.clone()
        Ss = S0s.clone()
        Mf = (S0f * Cin_fast).clone()
        Ms = (S0s * Cin_gw).clone()

        Sf_l, Ss_l, Qf_l, Qs_l, Quf_l = [], [], [], [], []
        Cf_l, Cs_l, floc_l, qloc_l = [], [], [], []
        fuf_l, ff_l, kf_l = [], [], []

        eps = 1e-6
        for t in range(T):
            Pe_t = precip[t]  # [n_sub]
            ET_t = et[t]

            Kf_t = self._Kf_of_t(Pe_t, Kf_recession, Kf_storm, Kf_a, Kf_b)

            f_uf, f_f = self.partition_net(
                sub_idx, Pe_t / pe_mean, ET_t / et_mean,
                sin_doy[t].expand(self.n_sub), cos_doy[t].expand(self.n_sub),
            )
            Puf = f_uf * Pe_t
            I = Pe_t - Puf
            If = f_f * I
            Rg = I - If

            ET_f = alphaET * ET_t
            ET_s = (1 - alphaET) * ET_t

            const_f = If - ET_f
            const_s = Rg - ET_s

            Qf = Kf_t * Sf
            Qs = Ks * Ss

            Sf_new = self._exp_step(Sf, Kf_t, const_f).clamp_min(eps)
            Ss_new = self._exp_step(Ss, Ks, const_s).clamp_min(eps)

            # Fast pathway (ultra-fast bypass + fast-reservoir infiltration)
            # carries fresh event-water chemistry; groundwater recharge
            # carries the higher, water-rock-contact-aged signature.
            mass_in_f = If * Cin_fast
            mass_in_s = Rg * Cin_gw
            Mf_new = self._exp_step(Mf, Kf_t, mass_in_f).clamp_min(0.0)
            Ms_new = self._exp_step(Ms, Ks, mass_in_s).clamp_min(0.0)

            Cf = (Mf / Sf.clamp_min(eps)).clamp(0.0, CF_CEILING_MGL)
            Cs = Ms / Ss.clamp_min(eps)

            Qloc = Puf + Qf + Qs                             # mm/day, local
            Floc = Puf * Cin_fast + Qf * Cf + Qs * Cs        # mm/day * mg/L, local mass flux

            Sf_l.append(Sf); Ss_l.append(Ss)
            Qf_l.append(Qf); Qs_l.append(Qs); Quf_l.append(Puf)
            Cf_l.append(Cf); Cs_l.append(Cs)
            qloc_l.append(Qloc); floc_l.append(Floc)
            fuf_l.append(f_uf); ff_l.append(f_f); kf_l.append(Kf_t)

            Sf, Ss, Mf, Ms = Sf_new, Ss_new, Mf_new, Ms_new

        Qloc = torch.stack(qloc_l)      # [T, n_sub] mm/day
        Floc = torch.stack(floc_l)      # [T, n_sub] mm/day * mg/L

        Qloc_vol = Qloc * self.area_cms      # [T, n_sub] m^3/s
        Floc_vol = Floc * self.area_cms      # [T, n_sub] (m^3/s)*(mg/L)

        # Routed totals: Qtot_j = Qloc_j + sum_{u in U(j)} Qtot_u
        # Solve node-by-node in topological order (headwaters first).
        Qtot_vol = Qloc_vol.clone()
        Ftot_vol = Floc_vol.clone()
        order = ["Rustlers", "Copper", "EAQ", "Gothic_ME", "PH_LT"]
        for b in order:
            j = SUB_IDX[b]
            for u in UPSTREAM[b]:
                ui = SUB_IDX[u]
                Qtot_vol[:, j] = Qtot_vol[:, j] + Qtot_vol[:, ui]
                Ftot_vol[:, j] = Ftot_vol[:, j] + Ftot_vol[:, ui]

        # C_tot's denominator gets its OWN floor, deliberately more
        # conservative than -- and decoupled from -- whatever Q_tot_vol
        # itself reports (Q_tot_vol above is returned to callers
        # completely unclamped; this floor only affects the division
        # below). Empirically calibrated against the two pre-fix
        # checkpoints (see README): the worst-case flux/floor ratio on any
        # day with Qtot<0.05 cms needed a floor of ~0.0079 cms to keep
        # C_tot under the ceiling below; CTOT_DENOM_FLOOR adds headroom on
        # top of that. CTOT_CEILING_MGL is a hard backstop independent of
        # whether the floor is sufficient: the max stream chloride ever
        # observed in this repo (1.8542 mg/L, PH_LT) x 5 safety margin.
        Ctot = Ftot_vol / Qtot_vol.clamp_min(CTOT_DENOM_FLOOR)
        Ctot = Ctot.clamp(0.0, CTOT_CEILING_MGL)

        return {
            "Sf": torch.stack(Sf_l), "Ss": torch.stack(Ss_l),
            "Qf": torch.stack(Qf_l), "Qs": torch.stack(Qs_l), "Quf": torch.stack(Quf_l),
            "Cf": torch.stack(Cf_l), "Cs": torch.stack(Cs_l),
            "Qloc_vol": Qloc_vol, "Qtot_vol": Qtot_vol, "Ctot": Ctot,
            "f_uf": torch.stack(fuf_l), "f_f": torch.stack(ff_l),
            "Kf_t": torch.stack(kf_l),  # [T, n_sub], the time-varying K_f,j(t)
            "Kf_recession": Kf_recession, "Kf_storm": Kf_storm,
            "Kf_a": Kf_a, "Kf_b": Kf_b, "Ks": Ks, "alphaET": alphaET,
            "Cin_fast": Cin_fast, "Cin_gw": Cin_gw,
        }
