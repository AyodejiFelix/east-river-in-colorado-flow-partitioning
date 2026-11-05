"""
Data loading and alignment for the East River tracer-informed PINN.

Loads the CLM-derived forcing (precipitation, ET), observed discharge, and
observed stream chloride for the five modeled sub-basin nodes (Rustlers
Gulch, Copper Creek, EAQ, Gothic_ME, PH_LT) and aligns them onto a common
daily date grid.

Forcing (precip_mm_day, ET_mm_day) is only available Dec 2013-Dec 2015
(CLM run window, per the corrected mathematical formulation doc's caveat in
Section 7). The mechanistic water/tracer balance can only be integrated
where forcing exists, so the training window is fixed to that period.

Effective water input: the governing equations define
P_e,j(t) = P_j(t) + SM_j(t) (rainfall + snowmelt). SM_j(t) is derived from
the Butte SNOTEL 380 station's daily SWE record as the day-to-day decline
in SWE (positive decrease = melt release to the surface; clipped at zero
on accumulation days, when SWE is building rather than releasing water).
Only a single SNOTEL point is available (not a per-sub-basin SWE product),
so the same SM(t) series is applied to all 5 sub-basins -- an explicit,
documented representativeness caveat (Butte 380 is one point standing in
for 5 distinct sub-basins spanning a range of elevations/aspects), not a
claim of spatial resolution the data doesn't have. This mirrors the doc's
own precedent of using pragmatic, stated simplifying assumptions where
finer-grained data isn't available (e.g. the single C_in,j(t) end-member).

Until 2026-08-09 the only available SNOTEL pull for this station covered
2023-10-01 to 2026-08-07, zero overlap with the Dec 2013-Dec 2015 training
window, so P_e,j = P_j (precip only) was used as a stand-in simplification.
An extended pull covering 2013-10-01 to 2026-08-08 (confirmed: all 737
training-window days present, no gaps) now supersedes that simplification.

Rain/snow phase partitioning: adding SM_j(t) on top of the RAW CLM
precipitation P_j(t) double-counts snowfall -- P_j(t) doesn't distinguish
rain from snow, so a day where precipitation falls as snow was being
treated as immediate liquid input AND (once it later melts) counted again
via SM_j(t). No temperature forcing exists anywhere in this repo (checked
every CSV; no temp/tair/tavg/tmin/tmax column anywhere) to use a standard
temperature-threshold rain/snow split, so the Butte 380 SWE record itself
is used as the proxy: on any day where SWE increases day-over-day (snow
accumulating, confirmed directly in the record, e.g. 2015-12-20->26 rises
2.9in->5.5in during exactly the event this fix targets), that day's
precipitation is withheld from the liquid input entirely and released
later, on whatever day SWE actually declines, via SM_j(t). On days where
SWE is flat or declining, precipitation passes through unchanged as
liquid input. This is the SWE-based proxy explicitly authorized as a
fallback when temperature forcing isn't available.

Melt-timing lag correction: PH_LT's simulated snowmelt peak landed
11-13 days early vs. observed in both 2014 and 2015 (see README's PH_LT
peak investigation). A no-retraining forward-pass test shifting SM_j(t)
forward by a range of offsets found a shared ~10-14 day lag brings both
years' peak timing close to zero offset simultaneously (confirmed
directly, not assumed) -- consistent with a single-point SNOTEL station
recording snow release earlier than it actually reaches these
sub-basins' streams (travel time / elevation-band melt-timing
differences a single point can't capture). MELT_LAG_DAYS=10 below
applies that correction: SM_j(t) is shifted forward 10 days (melt
recorded on day t at Butte 380 is assigned to day t+10 in the model's
P_e), zero-filled for the first 10 days of the record. This fixes WHEN
the peak occurs; it does NOT fix that PH_LT's 2015 peak magnitude is
too small (~37-39% of observed regardless of the lag applied, confirmed
directly) -- that remaining gap is attributed to the same single-station
representativeness limitation already documented for the ET forcing
(Rustlers/EAQ are 3.5-3.9km from their nearest matched met station; the
Butte 380 SWE station is similarly a single point standing in for melt
across 5 sub-basins of different elevation/aspect) and is treated here
as a characterized, accepted data limitation, not an open bug -- see
README's "PH_LT peak: known limitation" note.
"""
import numpy as np
import pandas as pd

SUBBASINS = ["Rustlers", "Copper", "EAQ", "Gothic_ME", "PH_LT"]

# Local (incremental) contributing area per modeled node, km^2.
# Rustlers/Copper/EAQ are their own GIS polygons. Gothic_ME's local area is
# EBC's own polygon plus the minor tributaries nested within it (Bradley,
# Rock, Quigley, Gothic, Avery, Marmot) per Section 4 of the formulation
# doc. PH_LT's local area is the Pumphouse polygon alone. Derived from
# subbasins_attributes.csv; sums to 84.6634 km^2, matching the doc's
# GIS cross-check against the published ~85 km^2 watershed area.
LOCAL_AREA_KM2 = {
    "Rustlers": 14.7838,
    "Copper": 23.6662,
    "EAQ": 5.2741,
    "Gothic_ME": 14.2675 + 2.5502 + 3.8227 + 3.5700 + 0.8966 + 0.5976 + 0.3773,
    "PH_LT": 14.8574,
}

# Upstream set U(j): which modeled nodes route into node j (Section 4).
UPSTREAM = {
    "Rustlers": [],
    "Copper": [],
    "EAQ": [],
    "Gothic_ME": ["Rustlers", "Copper", "EAQ"],
    "PH_LT": ["Gothic_ME"],
}

# mm/day over a local area -> m^3/s: area_km2 * 1e6 m2/km2 * 1e-3 m/mm / 86400 s/day
MM_PER_DAY_TO_CMS = {b: LOCAL_AREA_KM2[b] * 1e6 * 1e-3 / 86400.0 for b in SUBBASINS}


def _pivot(df, value_col, date_col="date", sub_col="sub_basin"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    piv = df.pivot_table(index=date_col, columns=sub_col, values=value_col, aggfunc="mean")
    return piv.reindex(columns=SUBBASINS)


SWE_FILE = "Butte_SNOTEL_380_SWE_SnowDepth_daily.csv"

# Melt-timing lag correction (see module docstring): a no-retraining
# forward-pass test found a shared 10-14 day forward shift of SM_j(t)
# brings both 2014 and 2015 PH_LT peak timing close to zero offset;
# 10 chosen as the offset that was directly tested and gave the
# smallest combined |timing offset| across both years (2014: -3d,
# 2015: 0d, vs. offset=14's +1d/+3d).
MELT_LAG_DAYS = 10


def _shift_forward(arr, offset):
    """arr[t] -> shifted[t] = arr[t-offset] (values released `offset`
    days later); zero-filled for t < offset. Works on [T] or [T,n_sub]."""
    if offset == 0:
        return arr.copy()
    shifted = np.zeros_like(arr)
    shifted[offset:] = arr[:-offset]
    return shifted


def _load_swe_delta_mm(data_dir, date_index):
    """Day-over-day change in SWE (mm, inches -> mm via x25.4) from the
    Butte 380 SNOTEL daily SWE record, reindexed to date_index. Positive =
    accumulating (snow day), negative = declining (melt day)."""
    swe = pd.read_csv(f"{data_dir}/{SWE_FILE}")
    swe["date"] = pd.to_datetime(swe["date"])
    swe = swe.set_index("date")["SWE_in"].sort_index()

    full_range = pd.date_range(date_index.min() - pd.Timedelta(days=1), date_index.max(), freq="D")
    swe = swe.reindex(full_range).ffill().bfill()
    swe_mm = swe * 25.4
    dswe_full = swe_mm.diff()
    return dswe_full.reindex(date_index).values.astype(np.float32)  # [T]


def _phase_partition(data_dir, date_index, precip_raw):
    """Splits raw CLM precipitation into a liquid (immediately routable)
    component and withholds the rest on days the Butte 380 SWE record
    shows accumulation (see module docstring for the proxy rationale).
    Returns (precip_liquid [T,n_sub], snowmelt_mm [T,n_sub], is_snow_day
    [T] bool) -- snowmelt_mm is SM_j(t) = max(0, -dSWE), shifted forward
    MELT_LAG_DAYS to correct the confirmed melt-timing lag (see module
    docstring); precip_liquid = 0 on is_snow_day, else = precip_raw.
    Same single-station series applied to all 5 sub-basins."""
    dswe_1d = _load_swe_delta_mm(data_dir, date_index)  # [T]
    is_snow_day = dswe_1d > 0.0  # [T] bool
    sm_1d = np.clip(-dswe_1d, 0.0, None).astype(np.float32)  # [T]
    sm_1d = _shift_forward(sm_1d, MELT_LAG_DAYS)  # melt-timing lag correction

    is_snow_day_2d = np.tile(is_snow_day[:, None], (1, len(SUBBASINS)))
    precip_liquid = np.where(is_snow_day_2d, 0.0, precip_raw).astype(np.float32)
    snowmelt_mm_day = np.tile(sm_1d[:, None], (1, len(SUBBASINS))).astype(np.float32)
    return precip_liquid, snowmelt_mm_day, is_snow_day


def load_dataset(data_dir="."):
    et = pd.read_csv(f"{data_dir}/ET_5subbasins_long.csv")
    precip = _pivot(et, "precip_mm_day")
    et_piv = _pivot(et, "ET_mm_day")

    date_index = pd.date_range(precip.index.min(), precip.index.max(), freq="D")
    precip = precip.reindex(date_index)
    et_piv = et_piv.reindex(date_index)
    # CLM forcing is complete for every sub-basin/day in-repo; guard against
    # any stray gap with a short forward/back fill rather than silently
    # feeding NaNs into the ODE integrator.
    precip = precip.ffill().bfill()
    et_piv = et_piv.ffill().bfill()

    q = pd.read_csv(f"{data_dir}/discharge_5subbasins_daily_long.csv")
    q_piv = _pivot(q, "Q_cms").reindex(date_index)

    cl = pd.read_csv(f"{data_dir}/chloride_5subbasins_long.csv")
    cl = cl[cl["outlier_flag"].astype(str) != "True"]
    cl_piv = _pivot(cl, "chloride_mgL").reindex(date_index)

    precip_raw = precip.values.astype(np.float32)
    precip_liquid, snowmelt_mm_day, is_snow_day = _phase_partition(data_dir, date_index, precip_raw)
    pe_mm_day = precip_liquid + snowmelt_mm_day  # P_e,j = P_liquid,j + SM_j (snow days withheld from P_liquid)

    return {
        "dates": date_index,
        "precip_mm_day": precip_raw,  # [T,5], RAW CLM precip (rain+snow undifferentiated), for reference only
        "precip_liquid_mm_day": precip_liquid,  # [T,5], precip_raw with snow days withheld (0 on is_snow_day)
        "snowmelt_mm_day": snowmelt_mm_day,  # [T,5], SM_j(t), single-station series broadcast to all 5
        "is_snow_day": is_snow_day,  # [T] bool, single-station snow-day flag (SWE increasing)
        "pe_mm_day": pe_mm_day,  # [T,5], P_e,j = P_liquid,j + SM_j -- the effective water input to the model
        "et_mm_day": et_piv.values.astype(np.float32),  # [T,5]
        "q_obs_cms": q_piv.values.astype(np.float32),  # [T,5], NaN where missing
        "cl_obs_mgL": cl_piv.values.astype(np.float32),  # [T,5], NaN where missing
        "subbasins": SUBBASINS,
    }


if __name__ == "__main__":
    d = load_dataset(".")
    print("date range:", d["dates"][0], "->", d["dates"][-1], "n_days=", len(d["dates"]))
    for i, b in enumerate(SUBBASINS):
        nq = np.isfinite(d["q_obs_cms"][:, i]).sum()
        ncl = np.isfinite(d["cl_obs_mgL"][:, i]).sum()
        print(f"{b:10s} area={LOCAL_AREA_KM2[b]:7.3f} km2  Qobs_days={nq:4d}  Clobs_days={ncl:4d}")
    sm = d["snowmelt_mm_day"][:, 0]
    print(f"\nSM(t) (single-station, mm/day): mean={sm.mean():.3f}  max={sm.max():.3f}  "
          f"days with SM>0: {(sm > 0).sum()}/{len(sm)}")
    n_snow_days = d["is_snow_day"].sum()
    print(f"is_snow_day: {n_snow_days}/{len(d['is_snow_day'])} days flagged as snow "
          f"(precip withheld that day)")
    p_raw_mean = d["precip_mm_day"][:, 0].mean()
    p_liq_mean = d["precip_liquid_mm_day"][:, 0].mean()
    pe_mean = d["pe_mm_day"][:, 0].mean()
    print(f"P_raw mean={p_raw_mean:.3f} mm/day  ->  P_liquid mean={p_liq_mean:.3f} mm/day "
          f"(withheld {100*(1-p_liq_mean/p_raw_mean):.1f}%)  ->  P_e (=P_liquid+SM) mean={pe_mean:.3f} mm/day")
