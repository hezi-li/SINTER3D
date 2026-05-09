import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from SINTER3D.model_spatial import Model


# ============================================================
# Global constants for the mouse brain dataset
# ============================================================

UM_PER_Z_UNIT = 200.0  # 1 z unit = 200 um, derived from c2c_dist
N_TOTAL = 35           # Total number of slices

slice_names = [
    '01A', '02A', '03A', '04B', '05A', '06B', '07A', '08B', '09A', '10B',
    '11A', '12B', '13A', '14B', '15A', '16B', '17A', '18B', '19A', '20B',
    '21A', '22B', '23A', '24B', '25A', '26B', '27A', '28B', '29A', '30B',
    '31A', '32B', '33A', '34B', '35A',
]


# ============================================================
# Part 1: Utility functions
# ============================================================

def build_slice_to_z(adata_st):
    """Build a mapping from slice index to z coordinate."""
    slice_to_z = {}
    for s in sorted(adata_st.obs['slice'].unique()):
        idx = adata_st.obs['slice'] == s
        z_val = adata_st.obsm['3D_coor'][idx, 2].mean()
        slice_to_z[int(s)] = float(z_val)
    return slice_to_z


def design_fixed_holdout(
    n_total,
    slice_to_z,
    slice_names=None,
    n_holdout=5,
    exclude_boundary=True,
):
    """
    Design a fixed hold-out test set.
    
    Rules:
    1. Exclude the first and last slices to avoid extrapolation
    2. Distribute selected slices evenly by z coordinate rather than by index
    3. Check batch balance when slice_names are provided
    """
    all_slices = list(range(n_total))
    
    if exclude_boundary:
        eligible = all_slices[1:-1]
    else:
        eligible = all_slices.copy()
    
    # Sample evenly by z coordinate rather than by index
    eligible_z = np.array([slice_to_z[s] for s in eligible])
    z_min, z_max = eligible_z.min(), eligible_z.max()
    target_z = np.linspace(z_min, z_max, n_holdout)
    
    holdout_slices = []
    used = set()
    for tz in target_z:
        # Find the nearest unselected eligible slice
        diffs = np.abs(eligible_z - tz)
        for idx in np.argsort(diffs):
            if eligible[idx] not in used:
                holdout_slices.append(eligible[idx])
                used.add(eligible[idx])
                break
    
    holdout_slices = sorted(holdout_slices)
    train_pool = [i for i in all_slices if i not in holdout_slices]
    
    print("=" * 70)
    print("Fixed hold-out test set design")
    print("=" * 70)
    print(f"Holdout slices (n={len(holdout_slices)}): {holdout_slices}")
    print(f"Holdout z positions (mum): {[round(slice_to_z[s] * UM_PER_Z_UNIT, 1) for s in holdout_slices]}")
    
    if slice_names is not None:
        batches = [slice_names[i][-1] for i in holdout_slices]
        n_A = sum(1 for b in batches if b == 'A')
        n_B = sum(1 for b in batches if b == 'B')
        print(f"Batch distribution: A={n_A}, B={n_B}")
        if abs(n_A - n_B) > 1:
            print("Warning: Batch imbalance detected; manual adjustment may be required")
    
    print(f"Training candidate pool size: {len(train_pool)}")
    return holdout_slices, train_pool


def select_train_by_interval(
    train_pool,
    interval,
    n_total,
    boundary_slices=None,
):
    """Sample the training set from train_pool by slice interval."""
    if boundary_slices is None:
        boundary_slices = []
    
    if interval == 1:
        train_slices = sorted(train_pool)
    else:
        ideal_positions = np.arange(0, n_total, interval)
        train_pool_arr = np.array(sorted(train_pool))
        train_slices = []
        for ideal in ideal_positions:
            nearest = train_pool_arr[np.argmin(np.abs(train_pool_arr - ideal))]
            train_slices.append(nearest)
        train_slices = sorted(set(train_slices))
    
    # Force inclusion of boundary slices
    for b in boundary_slices:
        if b in train_pool and b not in train_slices:
            train_slices.append(b)
    
    return sorted(set(train_slices))


def compute_z_distance_to_train(test_slices, train_slices, slice_to_z):
    """Compute the z-distance from each test slice to its nearest training slice, in z units."""
    train_z = np.array([slice_to_z[s] for s in train_slices])
    distances = {}
    for t in test_slices:
        t_z = slice_to_z[t]
        distances[t] = float(np.min(np.abs(train_z - t_z)))
    return pd.Series(distances, name="z_distance_to_train")


def summarize_z_intervals(train_slices, slice_to_z):
    """Compute z-spacing statistics between training slices."""
    train_z = sorted([slice_to_z[s] for s in train_slices])
    z_gaps = np.diff(train_z)
    return {
        "z_gap_mean": float(np.mean(z_gaps)),
        "z_gap_min": float(np.min(z_gaps)),
        "z_gap_max": float(np.max(z_gaps)),
        "z_gap_std": float(np.std(z_gaps)),
        "spacing_mean_um": float(np.mean(z_gaps)) * UM_PER_Z_UNIT,
        "spacing_min_um": float(np.min(z_gaps)) * UM_PER_Z_UNIT,
        "spacing_max_um": float(np.max(z_gaps)) * UM_PER_Z_UNIT,
    }


# ============================================================
# Part 2: Evaluation metrics
# ============================================================

def rowwise_cosine_mean(X1, X2, eps=1e-8):
    numerator = np.sum(X1 * X2, axis=1)
    denominator = np.linalg.norm(X1, axis=1) * np.linalg.norm(X2, axis=1)
    return np.nanmean(numerator / (denominator + eps))


def compare_matrices(X1, X2, mode="cellwise"):
    if issparse(X1):
        X1 = X1.toarray()
    if issparse(X2):
        X2 = X2.toarray()
    X1, X2 = np.asarray(X1), np.asarray(X2)
    
    results = {}
    if mode == "cellwise":
        results["Cosine_similarity"] = rowwise_cosine_mean(X1, X2)
        pearsons, spearmans = [], []
        for i in range(X1.shape[0]):
            if np.std(X1[i]) == 0 or np.std(X2[i]) == 0:
                pearsons.append(np.nan)
                spearmans.append(np.nan)
                continue
            pearsons.append(pearsonr(X1[i], X2[i])[0])
            spearmans.append(spearmanr(X1[i], X2[i])[0])
        results["Pearson_correlation"] = np.nanmean(pearsons)
        results["Spearman_correlation"] = np.nanmean(spearmans)
    
    results["MSE"] = float(np.mean((X1 - X2) ** 2))
    return results


def compare_by_slice(adata, key_true, key_pred, slice_key="slice", mode="cellwise"):
    results = {}
    for s in sorted(adata.obs[slice_key].unique()):
        idx = adata.obs[slice_key] == s
        results[s] = compare_matrices(
            adata.obsm[key_true][idx],
            adata.obsm[key_pred][idx],
            mode=mode,
        )
    return pd.DataFrame(results).T


# ============================================================
# Part 3: Complete workflow for a single interval
# ============================================================

def run_one_interval(
    interval,
    adata_st,
    adata_st_list,
    adata_basis,
    holdout_slices,
    train_pool,
    slice_to_z,
    config,
    boundary_slices=None,
    save_prefix="z_sampling",
):
    train_slices = select_train_by_interval(
        train_pool=train_pool,
        interval=interval,
        n_total=N_TOTAL,
        boundary_slices=boundary_slices,
    )
    
    distance_series = compute_z_distance_to_train(holdout_slices, train_slices, slice_to_z)
    z_summary = summarize_z_intervals(train_slices, slice_to_z)
    
    print("\n" + "=" * 70)
    print(f"Interval = {interval}")
    print(f"  Training slices (n={len(train_slices)}): {train_slices}")
    print(f"  Average training spacing: {z_summary['spacing_mean_um']:.1f} mum "
          f"(range {z_summary['spacing_min_um']:.0f} - {z_summary['spacing_max_um']:.0f} mum)")
    print(f"  Distance from holdout slices to nearest training slices (mum):")
    for s, d in distance_series.items():
        print(f"    slice {s}: {d * UM_PER_Z_UNIT:.0f} mum")
    
    # Build data subsets
    adata_st_list_train = [adata_st_list[i] for i in train_slices]
    adata_st_train = adata_st[adata_st.obs["slice"].isin(train_slices)].copy()
    adata_st_train.obs["type"] = "train"
    
    adata_st_test = adata_st[adata_st.obs["slice"].isin(holdout_slices)].copy()
    adata_st_test.obs["type"] = "test"
    
    # Train; important: pass the subset
    model = Model(
        adata_st_list_train,
        adata_st_train,
        adata_basis,
        slice_idx=train_slices,
        config=config,
    )
    model.train()
    
    # Inference
    adata_pred = model.inference_latent(adata_st_test, decode=True)
    
    if "X_origin" not in adata_pred.obsm:
        X_orig = adata_st_test.X
        if issparse(X_orig):
            X_orig = X_orig.toarray()
        adata_pred.obsm["X_origin"] = np.asarray(X_orig)
    
    # Global metrics
    global_result = compare_matrices(
        adata_pred.obsm["X_origin"],
        adata_pred.obsm["X_pred"],
        mode="cellwise",
    )
    global_result.update({
        "interval": interval,
        "n_train": len(train_slices),
        "n_holdout": len(holdout_slices),
        "train_slices": ",".join(map(str, train_slices)),
        **z_summary,
    })
    
    # Slice-level metrics
    slice_result = compare_by_slice(
        adata_pred, "X_origin", "X_pred", slice_key="slice", mode="cellwise",
    )
    slice_result["interval"] = interval
    slice_result["slice"] = slice_result.index.astype(int)
    slice_result["z_distance_to_train"] = slice_result["slice"].map(distance_series)
    slice_result["um_distance_to_train"] = slice_result["z_distance_to_train"] * UM_PER_Z_UNIT
    slice_result["n_train"] = len(train_slices)
    slice_result["spacing_mean_um"] = z_summary["spacing_mean_um"]
    
    # Save
    if os.path.dirname(save_prefix):
        os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    
    global_df = pd.DataFrame([global_result])
    global_df.to_csv(f"{save_prefix}_interval_{interval:02d}_global.csv", index=False)
    slice_result.to_csv(f"{save_prefix}_interval_{interval:02d}_by_slice.csv", index=False)
    
    print(f"  Cosine = {global_result['Cosine_similarity']:.4f}, "
          f"Pearson = {global_result['Pearson_correlation']:.4f}")
    
    return {
        "interval": interval,
        "global_df": global_df,
        "slice_df": slice_result,
        "train_slices": train_slices,
    }


