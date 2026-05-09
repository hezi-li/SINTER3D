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
    Generate virtual slices by using the nearest real slice as a full template, with a light reference correction applied only to boundary points.

    Parameters
    ----------
    boundary_ratio : float
        Start quantile for the boundary region. Larger values include fewer points in the correction.
        Recommended values: 0.80 or 0.85.
    lambda_max : float
        Maximum boundary correction strength. Keep this value small.
        Recommended range: 0.05-0.10.
    smooth_k : int
        Apply kNN smoothing to the boundary displacement field.
    max_shift_factor : float
        Limit the maximum displacement per point, expressed as a multiple of the nearest-neighbor center distance in the template slice.
        Recommended range: 0.5-1.0.
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

        # Use the nearest real slice as the template
        template_idx = int(np.argmin(np.abs(z_real - target_z)))
        template = records[template_idx]
        template_xy = template["xy"].copy()

        # Select a reference slice from the opposite or enclosing side
        if lower_idx == upper_idx:
            ref_idx = template_idx
        else:
            if template_idx == lower_idx:
                ref_idx = upper_idx
            elif template_idx == upper_idx:
                ref_idx = lower_idx
            else:
                # This should not happen; fall back to the nearer slice on the other side
                ref_idx = lower_idx if abs(z_real[lower_idx] - target_z) < abs(z_real[upper_idx] - target_z) else upper_idx

        ref = records[ref_idx]
        ref_xy = ref["xy"]

        # Initial coordinates: fully copy the template XY coordinates
        xy_final = template_xy.copy()

        # Apply a light boundary correction if a reference slice is available
        if ref_idx != template_idx and template_xy.shape[0] > 2 and ref_xy.shape[0] > 2:
            # Template nearest-neighbor center distance, used to cap maximum displacement
            tree_template = cKDTree(template_xy)
            nn_dists, _ = tree_template.query(template_xy, k=2)
            c2c = float(np.median(nn_dists[:, 1]))
            max_shift = max_shift_factor * c2c

            # Find nearest neighbors in the reference slice
            tree_ref = cKDTree(ref_xy)
            dist_ref, idx_ref = tree_ref.query(template_xy, k=1)
            matched_ref = ref_xy[idx_ref]

            # Raw boundary displacement
            disp = matched_ref - template_xy

            # Cap excessive displacement to avoid outliers or clustering
            disp_norm = np.linalg.norm(disp, axis=1)
            too_large = disp_norm > max_shift
            if np.any(too_large):
                scale = np.ones_like(disp_norm)
                scale[too_large] = max_shift / (disp_norm[too_large] + 1e-8)
                disp = disp * scale[:, None]

            # Smooth the displacement field
            disp_smooth = _smooth_vector_field(template_xy, disp, k=smooth_k)

            # Apply a small weight only to boundary points
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