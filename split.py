"""
Held-out train/test split for the 3-way comparative analysis (tracer PINN vs.
hydrology-only PINN vs. LSTM baseline).

No train/test split existed anywhere in this codebase before this file --
every prior checkpoint (results_meltlag_*, etc.) fit against 100% of the
observed record and reports in-sample metrics. That's fine for the project's
own diagnostic work, but is NOT a valid basis for a "does physics/does the
tracer help" comparison against an LSTM baseline: scoring one model in-sample
and another out-of-sample mechanically biases the comparison regardless of
which model is actually better.

Split design: a chronological cutoff was checked first and rejected --
Rustlers and Copper have ZERO discharge observations before 2015-07, so any
single date cutoff leaves those two gauges with no training signal at all.
Instead: a fixed-seed, STRATIFIED RANDOM 80/20 split of each gauge's own
observed days, done separately for Q and Cl (a day observed for Q isn't
necessarily observed for Cl, and vice versa). Forcing (P_e, ET) is never
masked -- it's a continuous physical input, not a target; only which
*observed* days feed the training loss vs. the held-out evaluation changes.

The same mask (same seed, computed once here) is reused, unmodified, across
all three models' training runs so the comparison is apples-to-apples.
"""
import numpy as np


HOLDOUT_SEED = 42
TRAIN_FRAC = 0.8


def build_holdout_masks(q_obs, cl_obs, seed=HOLDOUT_SEED, train_frac=TRAIN_FRAC):
    """q_obs, cl_obs: [T, n_sub] arrays (NaN where unobserved).
    Returns dict of boolean [T, n_sub] masks: q_train, q_test, cl_train, cl_test.
    q_train & q_test are disjoint and their union is exactly isfinite(q_obs)
    (same for cl). Split is done independently per gauge, per variable, so
    every gauge contributes both train and test observations regardless of
    how sparse it is.
    """
    rng = np.random.RandomState(seed)
    T, n_sub = q_obs.shape

    def split_one(obs):
        train_mask = np.zeros_like(obs, dtype=bool)
        test_mask = np.zeros_like(obs, dtype=bool)
        for j in range(n_sub):
            idx = np.where(np.isfinite(obs[:, j]))[0]
            idx = idx.copy()
            rng.shuffle(idx)
            n_train = int(round(len(idx) * train_frac))
            train_idx, test_idx = idx[:n_train], idx[n_train:]
            train_mask[train_idx, j] = True
            test_mask[test_idx, j] = True
        return train_mask, test_mask

    q_train, q_test = split_one(q_obs)
    cl_train, cl_test = split_one(cl_obs)
    return {"q_train": q_train, "q_test": q_test, "cl_train": cl_train, "cl_test": cl_test}


def save_masks(masks, path):
    np.savez(path, **{k: v for k, v in masks.items()})


def load_masks(path):
    npz = np.load(path)
    return {k: npz[k] for k in ["q_train", "q_test", "cl_train", "cl_test"]}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data import load_dataset, SUBBASINS

    d = load_dataset("/home/user/New/uploaded_data")
    masks = build_holdout_masks(d["q_obs_cms"], d["cl_obs_mgL"])
    save_masks(masks, "/home/user/New/results_comparative/holdout_split.npz")

    print(f"Held-out split (seed={HOLDOUT_SEED}, train_frac={TRAIN_FRAC}):")
    for i, b in enumerate(SUBBASINS):
        nq_tr, nq_te = masks["q_train"][:, i].sum(), masks["q_test"][:, i].sum()
        ncl_tr, ncl_te = masks["cl_train"][:, i].sum(), masks["cl_test"][:, i].sum()
        print(f"  {b:10s}  Q: train={nq_tr:4d} test={nq_te:4d}   "
              f"Cl: train={ncl_tr:4d} test={ncl_te:4d}")
    # sanity: train+test must equal total finite obs, exactly, no overlap
    assert np.array_equal(masks["q_train"] | masks["q_test"], np.isfinite(d["q_obs_cms"]))
    assert not np.any(masks["q_train"] & masks["q_test"])
    assert np.array_equal(masks["cl_train"] | masks["cl_test"], np.isfinite(d["cl_obs_mgL"]))
    assert not np.any(masks["cl_train"] & masks["cl_test"])
    print("sanity checks passed: train/test disjoint, union == all observed days")
