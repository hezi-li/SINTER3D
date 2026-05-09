import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection

import numpy as np
import pandas as pd
import anndata as ad
from scipy.spatial import cKDTree
from matplotlib.colors import Normalize



def find_edge_neighbors_visium(coords, c2c_dist, tolerance=0.3):
    """Find the six edge neighbors for each spot in a Visium hexagonal grid."""
    tree = cKDTree(coords)
    edge_lo = c2c_dist * (1 - tolerance)
    edge_hi = c2c_dist * (1 + tolerance)
    
    edge_neighbors = {}
    for i, c in enumerate(coords):
        dists, idxs = tree.query(c, k=8, distance_upper_bound=edge_hi)
        neighbors = []
        for d, j in zip(dists, idxs):
            if j == i or j == len(coords):
                continue
            if edge_lo <= d <= edge_hi:
                neighbors.append(j)
        edge_neighbors[i] = neighbors
    return edge_neighbors


def densify_visium_slice(
    adata,
    coor_key="spatial",
    c2c_dist=None,
    tolerance=0.3,
):
    """
    Densify a single Visium slice by inserting new spots at the midpoint of each adjacent spot pair.
    
    Parameters
    ----------
    adata : AnnData
        Single slice
    coor_key : str
        Coordinate storage key. If the coordinates are 3D, such as '3D_coor', the first two dimensions are used for densification.
    c2c_dist : float, optional
        Center-to-center distance. If None, infer it from nearest-neighbor distances.
    tolerance : float
        Distance tolerance for neighbor detection.
    
    Returns
    -------
    new_coords : (M, 2) array
        2D coordinates of newly inserted spots.
    void_info : pd.DataFrame
        Metadata for each newly inserted spot.
    """
    # Read coordinates; if they are 3D, automatically reduce them to 2D
    coords_full = np.asarray(adata.obsm[coor_key])
    if coords_full.shape[1] >= 2:
        coords = coords_full[:, :2]  # Use XY only
    else:
        raise ValueError(f"Insufficient coordinate dimensions: {coords_full.shape}")
    
    # Automatically infer c2c_dist
    if c2c_dist is None:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        c2c_dist = float(np.median(dists[:, 1]))
        print(f"  Automatically inferred c2c_dist = {c2c_dist:.2f}")
    
    edge_neighbors = find_edge_neighbors_visium(coords, c2c_dist, tolerance)
    
    n_neighbors_dist = pd.Series(
        [len(v) for v in edge_neighbors.values()]
    ).value_counts().sort_index()
    print(f"  Neighbor-count distribution: {dict(n_neighbors_dist)}")
    
    # Edge voids: midpoints of each neighbor pair, deduplicated
    void_pairs = set()
    for i, neighbors in edge_neighbors.items():
        for j in neighbors:
            void_pairs.add(frozenset([i, j]))
    
    new_coords = []
    void_info = []
    for k, pair in enumerate(void_pairs):
        i, j = sorted(pair)
        midpoint = (coords[i] + coords[j]) / 2
        new_coords.append(midpoint)
        void_info.append({
            "void_id": f"void_{k}",
            "src_spot1": adata.obs_names[i],
            "src_spot2": adata.obs_names[j],
            "src_idx1": i,
            "src_idx2": j,
            "x": midpoint[0],
            "y": midpoint[1],
        })
    
    new_coords = np.array(new_coords)
    void_info_df = pd.DataFrame(void_info).set_index("void_id")
    
    print(f"  Number of original spots: {len(coords)}")
    print(f"  Number of inserted voids: {len(new_coords)}")
    print(f"  Densification factor: {(len(coords) + len(new_coords)) / len(coords):.2f}x")
    
    return new_coords, void_info_df


def build_densified_adata(
    adata_original,
    new_coords,
    coor_key="spatial",
    coor_3d_key="3D_coor",
    z_value=None,
    inherit_obs_cols=("slice",),
):
    """
    Merge newly inserted spots into the original adata object.
    
    Supports two coordinate cases:
    - coor_key contains 2D coordinates
    - coor_key contains 3D coordinates, such as '3D_coor' with xyz columns
    
    Parameters
    ----------
    inherit_obs_cols : tuple of str
        obs columns that virtual spots should inherit from the original adata when values match the original slice.
        By default, inherit 'slice' because all virtual spots within the same slice belong to that slice.
        Other columns remain NaN.
    """
    n_new = len(new_coords)
    
    # ------------------------------------------------------------
    # Expression matrix: fill all virtual spots with NaN; these are predicted by the model and should be replaced with 0 before inference
    # ------------------------------------------------------------
    X_new = np.full((n_new, adata_original.n_vars), np.nan, dtype=np.float32)
    
    # ------------------------------------------------------------
    # obs: mark virtual spots, inherit specified columns, and fill remaining values with NaN
    # ------------------------------------------------------------
    obs_new = pd.DataFrame(index=[f"virtual_{i}" for i in range(n_new)])
    for col in adata_original.obs.columns:
        # Key fix: for inherited columns such as 'slice', use values from the original slice
        if col in inherit_obs_cols and col in adata_original.obs.columns:
            unique_vals = adata_original.obs[col].unique()
            if len(unique_vals) == 1:
                # If the column has the same value across the slice, virtual spots inherit it directly
                obs_new[col] = unique_vals[0]
            else:
                # Exception: if the column has non-unique values in the original slice, fall back to NaN and warn
                print(f"  Warning: Column '{col}' has non-unique values in the original slice; fill this column with NaN for virtual spots")
                obs_new[col] = np.nan
        else:
            obs_new[col] = np.nan
    obs_new["spot_type"] = "virtual"
    
    obs_orig = adata_original.obs.copy()
    obs_orig["spot_type"] = "original"
    
    # ------------------------------------------------------------
    # Process obsm
    # ------------------------------------------------------------
    obsm_new = {}
    
    orig_coor = np.asarray(adata_original.obsm[coor_key])
    if orig_coor.shape[1] == 2:
        obsm_new[coor_key] = np.vstack([orig_coor, new_coords])
    elif orig_coor.shape[1] == 3:
        if z_value is None:
            z_value = float(orig_coor[:, 2].mean())
        new_3d = np.column_stack([new_coords, np.full(n_new, z_value)])
        obsm_new[coor_key] = np.vstack([orig_coor, new_3d])
    
    # If a separate 3D coordinate key exists and differs from coor_key
    if coor_3d_key is not None and coor_3d_key != coor_key:
        if coor_3d_key in adata_original.obsm:
            orig_3d = np.asarray(adata_original.obsm[coor_3d_key])
            if z_value is None:
                z_value = float(orig_3d[:, 2].mean())
            new_3d = np.column_stack([new_coords, np.full(n_new, z_value)])
            obsm_new[coor_3d_key] = np.vstack([orig_3d, new_3d])
    
    # ------------------------------------------------------------
    # Concatenate .X and construct AnnData
    # ------------------------------------------------------------
    X_orig = adata_original.X.toarray() if hasattr(adata_original.X, 'toarray') else adata_original.X
    
    adata_dense = ad.AnnData(
        X=np.vstack([X_orig, X_new]),
        obs=pd.concat([obs_orig, obs_new], axis=0),
        var=adata_original.var.copy(),
        obsm=obsm_new,
    )
    
    return adata_dense

def visualize_densification(
    coords_orig,
    coords_new,
    spot_radius=None,
    c2c_dist=None,
    save_path=None,
):
    # Automatic inference
    if c2c_dist is None:
        tree = cKDTree(coords_orig)
        dists, _ = tree.query(coords_orig, k=2)
        c2c_dist = float(np.median(dists[:, 1]))
    
    # Key change 1: use the true physical radius (c2c * 0.275) instead of 0.35
    if spot_radius is None:
        spot_radius = c2c_dist * 0.275
    
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.4], wspace=0.25)
    
    # ============================================================
    # Panel A remains unchanged
    # ============================================================
    ax_a = fig.add_subplot(gs[0])
    ax_a.scatter(coords_orig[:, 0], coords_orig[:, 1], 
                 s=4, c='#2E86AB', alpha=0.8, label=f'Original ({len(coords_orig)})')
    ax_a.scatter(coords_new[:, 0], coords_new[:, 1],
                 s=2, c='#E63946', alpha=0.6, label=f'Inserted ({len(coords_new)})')
    ax_a.set_aspect('equal')
    ax_a.invert_yaxis()
    ax_a.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax_a.set_title('A. Global tissue coverage', fontsize=13, fontweight='bold')
    ax_a.set_xlabel('x (pixels)')
    ax_a.set_ylabel('y (pixels)')
    
    center_idx = len(coords_orig) // 2
    cx, cy = coords_orig[center_idx]
    
    box_b_size = c2c_dist * 1.5
    rect_b = plt.Rectangle((cx - box_b_size, cy - box_b_size),
                            2 * box_b_size, 2 * box_b_size,
                            fill=False, edgecolor='orange', linewidth=2.5)
    ax_a.add_patch(rect_b)
    ax_a.text(cx + box_b_size, cy - box_b_size, 'B', 
              fontsize=14, fontweight='bold', color='orange',
              verticalalignment='bottom')
    
    box_c_size = c2c_dist * 4
    rect_c = plt.Rectangle((cx - box_c_size, cy - box_c_size),
                            2 * box_c_size, 2 * box_c_size,
                            fill=False, edgecolor='green', linewidth=2.5)
    ax_a.add_patch(rect_c)
    ax_a.text(cx + box_c_size, cy - box_c_size, 'C',
              fontsize=14, fontweight='bold', color='green',
              verticalalignment='bottom')
    
    # ============================================================
    # Panel B: key change -- draw virtual spots as hollow circles plus crosses
    # ============================================================
    ax_b = fig.add_subplot(gs[1])
    
    tree = cKDTree(coords_orig)
    dists, idxs = tree.query([cx, cy], k=8)
    neighbor_idxs = []
    for d, i in zip(dists[1:], idxs[1:]):
        if d <= c2c_dist * 1.3:
            neighbor_idxs.append(i)
    
    relevant_voids = []
    spots_of_interest = [(cx, cy)] + [tuple(coords_orig[i]) for i in neighbor_idxs]
    for nv in coords_new:
        for sp in spots_of_interest:
            if np.hypot(nv[0] - sp[0], nv[1] - sp[1]) < c2c_dist * 0.6:
                relevant_voids.append(nv)
                break
    relevant_voids = np.array(relevant_voids) if relevant_voids else np.empty((0, 2))
    
    # Center spot: solid dark blue
    circle_center = Circle((cx, cy), spot_radius, 
                            facecolor='#1A4F72', edgecolor='black',
                            linewidth=1.5, alpha=0.95, zorder=3)
    ax_b.add_patch(circle_center)
    ax_b.text(cx, cy, 'A', ha='center', va='center', 
              color='white', fontsize=11, fontweight='bold', zorder=4)
    
    # Six neighbors: solid light blue
    for k, i in enumerate(neighbor_idxs):
        nx, ny = coords_orig[i]
        circle = Circle((nx, ny), spot_radius,
                        facecolor='#5DADE2', edgecolor='black',
                        linewidth=1.0, alpha=0.85, zorder=3)
        ax_b.add_patch(circle)
        ax_b.text(nx, ny, f'N{k+1}', ha='center', va='center',
                  color='white', fontsize=9, fontweight='bold', zorder=4)
        # Line from A to N; solid line indicates neighbor relationship
        ax_b.plot([cx, nx], [cy, ny], '-', color='#888', 
                  alpha=0.5, linewidth=1.0, zorder=1)
    
    # Key change: draw virtual spots as hollow red circles plus red dots
    for vx, vy in relevant_voids:
        # Hollow red circle with dashed outline: indicates the covered area without conflicting visually with solid blue circles
        circle_v = Circle((vx, vy), spot_radius,
                          facecolor='none', edgecolor='#E63946',
                          linewidth=1.8, alpha=0.9, zorder=2,
                          linestyle='--')
        ax_b.add_patch(circle_v)
        # Small red dot at the center: marks the spot center position
        ax_b.plot(vx, vy, 'o', color='#E63946',
                  markersize=4, zorder=4)
    
    ax_b.set_xlim(cx - box_b_size, cx + box_b_size)
    ax_b.set_ylim(cy - box_b_size, cy + box_b_size)
    ax_b.set_aspect('equal')
    ax_b.invert_yaxis()
    ax_b.set_title('B. Geometry: 1 spot + 6 neighbors + 6 inserted voids',
                   fontsize=13, fontweight='bold')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A4F72',
               markersize=12, markeredgecolor='black', label='Original spot'),
        Line2D([0], [0], marker='o', color='#E63946', markerfacecolor='none',
               markersize=12, markeredgewidth=1.8, linestyle='--',
               label='Virtual spot'),
    ]
    ax_b.legend(handles=legend_elements, loc='upper left', fontsize=9,
                framealpha=0.95)
    
    # ============================================================
    # Panel C: also use hollow circles to distinguish virtual spots
    # ============================================================
    ax_c = fig.add_subplot(gs[2])
    
    mask_orig = (
        (coords_orig[:, 0] > cx - box_c_size) & (coords_orig[:, 0] < cx + box_c_size) &
        (coords_orig[:, 1] > cy - box_c_size) & (coords_orig[:, 1] < cy + box_c_size)
    )
    mask_new = (
        (coords_new[:, 0] > cx - box_c_size) & (coords_new[:, 0] < cx + box_c_size) &
        (coords_new[:, 1] > cy - box_c_size) & (coords_new[:, 1] < cy + box_c_size)
    )
    
    coords_orig_sub = coords_orig[mask_orig]
    coords_new_sub = coords_new[mask_new]
    
    # Original spots: solid circles
    for x, y in coords_orig_sub:
        circle = Circle((x, y), spot_radius,
                        facecolor='#2E86AB', edgecolor='black',
                        linewidth=0.6, alpha=0.9, zorder=3)
        ax_c.add_patch(circle)
    
    # Virtual spots: hollow red circles
    for x, y in coords_new_sub:
        circle = Circle((x, y), spot_radius,
                        facecolor='none', edgecolor='#E63946',
                        linewidth=1.0, alpha=0.85, zorder=2,
                        linestyle='--')
        ax_c.add_patch(circle)
    
    ax_c.set_xlim(cx - box_c_size, cx + box_c_size)
    ax_c.set_ylim(cy - box_c_size, cy + box_c_size)
    ax_c.set_aspect('equal')
    ax_c.invert_yaxis()
    ax_c.set_title(
        f'C. Hexagonal lattice ({len(coords_orig_sub)} originals + {len(coords_new_sub)} voids)',
        fontsize=13, fontweight='bold')
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    
    plt.suptitle(
        f'Visium Slice Densification: {len(coords_orig)} -> {len(coords_orig) + len(coords_new)} spots '
        f'({(len(coords_orig) + len(coords_new))/len(coords_orig):.2f}x density)',
        fontsize=15, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()


def plot_gene_super_resolution(
    adata_super,
    slice_id,
    gene,
    coor_key='3D_coor',
    cmap='viridis',
    save_path=None,
):
    """
    Single slice and single gene: true expression in original spots versus predicted expression in all densified spots.
    """
    # Extract the slice
    adata_slice = adata_super[adata_super.obs['slice'] == slice_id].copy()
    
    is_orig = (adata_slice.obs['spot_type'] == 'original').values
    coords = adata_slice.obsm[coor_key][:, :2]
    
    # Find the gene index
    if gene not in adata_slice.var_names:
        print(f"  Warning: {gene} is not in var_names; skipping")
        return
    gene_idx = adata_slice.var_names.get_loc(gene)
    
    # True expression of original spots from .X
    X_full = adata_slice.X
    if hasattr(X_full, 'toarray'):
        X_full = X_full.toarray()
    expr_real_orig = X_full[is_orig, gene_idx]
    
    # Predicted expression of all spots from obsm['X_pred']
    expr_pred_all = adata_slice.obsm['X_pred'][:, gene_idx]
    
    # Set the color scale using true expression
    vmin = float(np.percentile(expr_real_orig, 2))
    vmax = float(np.percentile(expr_real_orig, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: original
    sc1 = axes[0].scatter(
        coords[is_orig, 0], coords[is_orig, 1],
        c=expr_real_orig, cmap=cmap, norm=norm,
        s=18, edgecolors='none',
    )
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    axes[0].set_title(
        f'Original  ({is_orig.sum()} spots)',
        fontsize=13, fontweight='bold',
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    plt.colorbar(sc1, ax=axes[0], shrink=0.7, label=f'{gene}')
    
    # Right: densified prediction
    sc2 = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=expr_pred_all, cmap=cmap, norm=norm,
        s=6, edgecolors='none',
    )
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[1].set_title(
        f'Super-resolution  ({len(coords)} spots, '
        f'{len(coords)/is_orig.sum():.1f}x density)',
        fontsize=13, fontweight='bold',
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    plt.colorbar(sc2, ax=axes[1], shrink=0.7, label=f'{gene} (predicted)')
    
    plt.suptitle(
        f'Slice {slice_id} - {gene} expression',
        fontsize=15, fontweight='bold', y=1.02,
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Save: {save_path}")
    plt.show()


def plot_all_slices_one_gene(
    adata_st,
    adata_super,
    gene,
    slice_ids=[0, 1, 2, 3],
    slice_names=None,
    coor_key='3D_coor',
    cmap='viridis',
    save_path=None,
):
    """
    Compare original versus densified results for one gene across multiple slices using an N x 2 grid.
    """
    if slice_names is None:
        slice_names = [f'Slice {sid}' for sid in slice_ids]
    
    if gene not in adata_super.var_names:
        print(f"  Warning: {gene} is not in var_names; skipping")
        return
    
    gene_idx = adata_super.var_names.get_loc(gene)
    
    # Global color scale based on true expression across all slices
    X_st = adata_st.X
    if hasattr(X_st, 'toarray'):
        X_st = X_st.toarray()
    all_expr = X_st[:, gene_idx]
    vmin = float(np.percentile(all_expr, 2))
    vmax = float(np.percentile(all_expr, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(len(slice_ids), 2, figsize=(12, 4 * len(slice_ids)))
    if len(slice_ids) == 1:
        axes = axes.reshape(1, -1)
    
    for row, sid in enumerate(slice_ids):
        # Original
        adata_orig_i = adata_st[adata_st.obs['slice'] == sid]
        coords_orig = adata_orig_i.obsm[coor_key][:, :2]
        X_i = adata_orig_i.X
        if hasattr(X_i, 'toarray'):
            X_i = X_i.toarray()
        expr_orig = X_i[:, gene_idx]
        
        # Densified prediction
        adata_super_i = adata_super[adata_super.obs['slice'] == sid]
        coords_super = adata_super_i.obsm[coor_key][:, :2]
        expr_super = adata_super_i.obsm['X_pred'][:, gene_idx]
        
        # Left column: original
        ax_l = axes[row, 0]
        ax_l.scatter(
            coords_orig[:, 0], coords_orig[:, 1],
            c=expr_orig, cmap=cmap, norm=norm, s=15, edgecolors='none',
        )
        ax_l.set_aspect('equal')
        ax_l.invert_yaxis()
        ax_l.set_xticks([])
        ax_l.set_yticks([])
        if row == 0:
            ax_l.set_title('Original', fontsize=13, fontweight='bold')
        ax_l.set_ylabel(slice_names[row], fontsize=12, fontweight='bold')
        
        # Right column: densified
        ax_r = axes[row, 1]
        sc_r = ax_r.scatter(
            coords_super[:, 0], coords_super[:, 1],
            c=expr_super, cmap=cmap, norm=norm, s=6, edgecolors='none',
        )
        ax_r.set_aspect('equal')
        ax_r.invert_yaxis()
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if row == 0:
            ax_r.set_title('Super-resolution', fontsize=13, fontweight='bold')
    
    # Shared colorbar
    fig.colorbar(sc_r, ax=axes, shrink=0.5, label=gene, location='right', pad=0.02)
    
    plt.suptitle(f'{gene} across {len(slice_ids)} DLPFC slices',
                 fontsize=15, fontweight='bold', y=1.005)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Save: {save_path}")
    plt.show()