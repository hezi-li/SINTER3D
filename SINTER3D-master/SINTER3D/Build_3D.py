import numpy as np
from scipy.spatial import cKDTree

def _get_slice_records(adata, spatial_name="3D_coor", slice_key="slice"):
    coords_3d = np.asarray(adata.obsm[spatial_name])
    slice_labels = adata.obs[slice_key].to_numpy()

    records = []
    for label in np.unique(slice_labels):
        mask = slice_labels == label
        coords_i = coords_3d[mask]
        records.append({
            "label": label,
            "z": float(np.mean(coords_i[:, 2])),
            "xy": coords_i[:, :2].copy(),
            "n": coords_i.shape[0],
        })

    records = sorted(records, key=lambda x: x["z"])
    return records


def _find_bracketing_slices(records, target_z):
    z_real = np.array([r["z"] for r in records], dtype=float)
    target_z = float(target_z)

    lower = np.where(z_real <= target_z)[0]
    upper = np.where(z_real >= target_z)[0]

    if len(lower) == 0:
        lower_idx = int(np.argmin(np.abs(z_real - target_z)))
    else:
        lower_idx = int(lower[-1])

    if len(upper) == 0:
        upper_idx = int(np.argmin(np.abs(z_real - target_z)))
    else:
        upper_idx = int(upper[0])

    return lower_idx, upper_idx


def _boundary_weight(xy, boundary_ratio=0.80, lambda_max=0.10):
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    rmax = np.max(radii)

    if rmax <= 1e-8:
        return np.zeros(xy.shape[0], dtype=float)

    w = (radii / rmax - boundary_ratio) / (1.0 - boundary_ratio)
    w = np.clip(w, 0.0, 1.0)

    return lambda_max * w


def _smooth_vector_field(xy, vectors, k=8):
    if k is None or k <= 1 or xy.shape[0] <= k:
        return vectors

    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k)

    return vectors[idx].mean(axis=1)


def generate_virtual_slice_template_boundary_refined(
    adata,
    target_z_list,
    spatial_name="3D_coor",
    slice_key="slice",
    boundary_ratio=0.80,
    lambda_max=0.08,
    smooth_k=8,
    max_shift_factor=1.0,
    verbose=True,
):
    """
    虚拟切片生成：最近真实切片作为完整模板 + 只对边界点进行轻微参考校正。

    Parameters
    ----------
    boundary_ratio : float
        边界区域起始比例。越大，参与校正的点越少。
        推荐 0.80 或 0.85。
    lambda_max : float
        最大边界校正强度。不要太大。
        推荐 0.05–0.10。
    smooth_k : int
        对边界位移场进行 kNN 平滑。
    max_shift_factor : float
        限制单点最大位移，单位为模板切片最近邻中心距倍数。
        推荐 0.5–1.0。
    """

    if not isinstance(target_z_list, (list, tuple, np.ndarray)):
        target_z_list = [target_z_list]

    records = _get_slice_records(
        adata,
        spatial_name=spatial_name,
        slice_key=slice_key
    )

    z_real = np.array([r["z"] for r in records], dtype=float)

    if verbose:
        print("Real z values:")
        print(z_real)

    virtual_all = []

    for target_z in target_z_list:
        target_z = float(target_z)

        lower_idx, upper_idx = _find_bracketing_slices(records, target_z)

        # 最近真实切片作为模板
        template_idx = int(np.argmin(np.abs(z_real - target_z)))
        template = records[template_idx]
        template_xy = template["xy"].copy()

        # 选择另一侧或包围侧参考切片
        if lower_idx == upper_idx:
            ref_idx = template_idx
        else:
            if template_idx == lower_idx:
                ref_idx = upper_idx
            elif template_idx == upper_idx:
                ref_idx = lower_idx
            else:
                # 理论上不会发生，兜底选另一侧更近的
                ref_idx = lower_idx if abs(z_real[lower_idx] - target_z) < abs(z_real[upper_idx] - target_z) else upper_idx

        ref = records[ref_idx]
        ref_xy = ref["xy"]

        # 初始坐标：完整复制模板 XY
        xy_final = template_xy.copy()

        # 如果有参考切片，进行边界轻微校正
        if ref_idx != template_idx and template_xy.shape[0] > 2 and ref_xy.shape[0] > 2:
            # 模板最近邻中心距，用于限制最大位移
            tree_template = cKDTree(template_xy)
            nn_dists, _ = tree_template.query(template_xy, k=2)
            c2c = float(np.median(nn_dists[:, 1]))
            max_shift = max_shift_factor * c2c

            # 找参考切片最近邻
            tree_ref = cKDTree(ref_xy)
            dist_ref, idx_ref = tree_ref.query(template_xy, k=1)
            matched_ref = ref_xy[idx_ref]

            # 原始边界位移
            disp = matched_ref - template_xy

            # 限制过大位移，避免外飞或聚团
            disp_norm = np.linalg.norm(disp, axis=1)
            too_large = disp_norm > max_shift
            if np.any(too_large):
                scale = np.ones_like(disp_norm)
                scale[too_large] = max_shift / (disp_norm[too_large] + 1e-8)
                disp = disp * scale[:, None]

            # 平滑位移场
            disp_smooth = _smooth_vector_field(template_xy, disp, k=smooth_k)

            # 只对边界点施加小权重
            lam = _boundary_weight(
                template_xy,
                boundary_ratio=boundary_ratio,
                lambda_max=lambda_max
            )

            xy_final = template_xy + lam[:, None] * disp_smooth

            if verbose:
                print(
                    f"target_z={target_z:g}: template z={template['z']:g} n={template['n']}, "
                    f"ref z={ref['z']:g} n={ref['n']}, "
                    f"boundary_ratio={boundary_ratio}, lambda_max={lambda_max}, "
                    f"max_shift={max_shift:.2f}"
                )
        else:
            if verbose:
                print(
                    f"target_z={target_z:g}: template z={template['z']:g} n={template['n']}, "
                    f"no boundary refinement"
                )

        xyz = np.column_stack([
            xy_final,
            np.full(xy_final.shape[0], target_z, dtype=float)
        ])

        virtual_all.append(xyz)

    return np.vstack(virtual_all)