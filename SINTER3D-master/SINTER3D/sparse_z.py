import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import issparse
from SINTER3D.model_spatial import Model


# ============================================================
# 全局常量（mouse brain 数据特定）
# ============================================================

UM_PER_Z_UNIT = 200.0  # 1 z单位 = 200 μm（来自 c2c_dist）
N_TOTAL = 35           # 总切片数

slice_names = [
    '01A', '02A', '03A', '04B', '05A', '06B', '07A', '08B', '09A', '10B',
    '11A', '12B', '13A', '14B', '15A', '16B', '17A', '18B', '19A', '20B',
    '21A', '22B', '23A', '24B', '25A', '26B', '27A', '28B', '29A', '30B',
    '31A', '32B', '33A', '34B', '35A',
]


# ============================================================
# Part 1: 工具函数
# ============================================================

def build_slice_to_z(adata_st):
    """构造 slice index → z 坐标 的映射。"""
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
    设计固定 hold-out 测试集。
    
    规则：
    1. 排除首尾切片避免外推
    2. 在剩余切片中按 z 坐标均匀分布（不是按 index 均匀）
    3. 检查批次平衡（如有 slice_names）
    """
    all_slices = list(range(n_total))
    
    if exclude_boundary:
        eligible = all_slices[1:-1]
    else:
        eligible = all_slices.copy()
    
    # 按 z 坐标均匀采样（而不是按 index 均匀）
    eligible_z = np.array([slice_to_z[s] for s in eligible])
    z_min, z_max = eligible_z.min(), eligible_z.max()
    target_z = np.linspace(z_min, z_max, n_holdout)
    
    holdout_slices = []
    used = set()
    for tz in target_z:
        # 找最近的、未被选过的 eligible slice
        diffs = np.abs(eligible_z - tz)
        for idx in np.argsort(diffs):
            if eligible[idx] not in used:
                holdout_slices.append(eligible[idx])
                used.add(eligible[idx])
                break
    
    holdout_slices = sorted(holdout_slices)
    train_pool = [i for i in all_slices if i not in holdout_slices]
    
    print("=" * 70)
    print("固定 Hold-out 测试集设计")
    print("=" * 70)
    print(f"Holdout slices (n={len(holdout_slices)}): {holdout_slices}")
    print(f"Holdout z 位置 (μm): {[round(slice_to_z[s] * UM_PER_Z_UNIT, 1) for s in holdout_slices]}")
    
    if slice_names is not None:
        batches = [slice_names[i][-1] for i in holdout_slices]
        n_A = sum(1 for b in batches if b == 'A')
        n_B = sum(1 for b in batches if b == 'B')
        print(f"批次分布: A={n_A}, B={n_B}")
        if abs(n_A - n_B) > 1:
            print("⚠️  批次不平衡，可能需要手动调整")
    
    print(f"训练候选池大小: {len(train_pool)}")
    return holdout_slices, train_pool


def select_train_by_interval(
    train_pool,
    interval,
    n_total,
    boundary_slices=None,
):
    """从 train_pool 中按 slice interval 抽训练集。"""
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
    
    # 强制加入边界
    for b in boundary_slices:
        if b in train_pool and b not in train_slices:
            train_slices.append(b)
    
    return sorted(set(train_slices))


def compute_z_distance_to_train(test_slices, train_slices, slice_to_z):
    """每张测试切片到最近训练切片的 z 距离（z 单位）。"""
    train_z = np.array([slice_to_z[s] for s in train_slices])
    distances = {}
    for t in test_slices:
        t_z = slice_to_z[t]
        distances[t] = float(np.min(np.abs(train_z - t_z)))
    return pd.Series(distances, name="z_distance_to_train")


def summarize_z_intervals(train_slices, slice_to_z):
    """训练切片之间的 z 间距统计。"""
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
# Part 2: 评估指标
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
# Part 3: 单个 interval 完整流程
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
    print(f"  训练切片 (n={len(train_slices)}): {train_slices}")
    print(f"  平均训练间距: {z_summary['spacing_mean_um']:.1f} μm "
          f"(范围 {z_summary['spacing_min_um']:.0f} - {z_summary['spacing_max_um']:.0f} μm)")
    print(f"  Holdout 切片到最近训练切片距离 (μm):")
    for s, d in distance_series.items():
        print(f"    slice {s}: {d * UM_PER_Z_UNIT:.0f} μm")
    
    # 构造数据子集
    adata_st_list_train = [adata_st_list[i] for i in train_slices]
    adata_st_train = adata_st[adata_st.obs["slice"].isin(train_slices)].copy()
    adata_st_train.obs["type"] = "train"
    
    adata_st_test = adata_st[adata_st.obs["slice"].isin(holdout_slices)].copy()
    adata_st_test.obs["type"] = "test"
    
    # 训练（关键：传子集！）
    model = Model(
        adata_st_list_train,
        adata_st_train,
        adata_basis,
        slice_idx=train_slices,
        config=config,
    )
    model.train()
    
    # 推理
    adata_pred = model.inference_latent(adata_st_test, decode=True)
    
    if "X_origin" not in adata_pred.obsm:
        X_orig = adata_st_test.X
        if issparse(X_orig):
            X_orig = X_orig.toarray()
        adata_pred.obsm["X_origin"] = np.asarray(X_orig)
    
    # 全局指标
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
    
    # 切片级指标
    slice_result = compare_by_slice(
        adata_pred, "X_origin", "X_pred", slice_key="slice", mode="cellwise",
    )
    slice_result["interval"] = interval
    slice_result["slice"] = slice_result.index.astype(int)
    slice_result["z_distance_to_train"] = slice_result["slice"].map(distance_series)
    slice_result["um_distance_to_train"] = slice_result["z_distance_to_train"] * UM_PER_Z_UNIT
    slice_result["n_train"] = len(train_slices)
    slice_result["spacing_mean_um"] = z_summary["spacing_mean_um"]
    
    # 保存
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


