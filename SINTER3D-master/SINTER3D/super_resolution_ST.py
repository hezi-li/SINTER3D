import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree


# ============================================================
# Part 1: Neighbor search and densification
# ============================================================

def find_edge_neighbors_st(coords, c2c_dist, tolerance=0.3):
    """Find the four edge neighbors (up, down, left, right) for each spot in an ST square grid."""
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


def find_corner_quads_st(coords, edge_neighbors, c2c_dist, tolerance=0.3):
    """
    Find each four-spot square in an ST square grid.
    
    Strategy: for each spot i, find two edge neighbors j and k such that j-k forms a diagonal direction.
    Then find the fourth spot that is an edge neighbor of both j and k and is not i; these four spots form a square.
    
    Returns
    -------
    quads : list of tuples
        Each element is a 4-tuple (i, j, l, k) containing the sorted, deduplicated spot indices of one square.
    """
    diag_dist = c2c_dist * np.sqrt(2)
    diag_lo = diag_dist * (1 - tolerance)
    diag_hi = diag_dist * (1 + tolerance)
    
    tree = cKDTree(coords)
    
    quads_set = set()
    
    for i in range(len(coords)):
        # Find diagonal neighbors of i with distance approximately sqrt(2) * c2c
        dists, idxs = tree.query(coords[i], k=10, distance_upper_bound=diag_hi)
        diag_candidates = []
        for d, j in zip(dists, idxs):
            if j == i or j == len(coords):
                continue
            if diag_lo <= d <= diag_hi:
                diag_candidates.append(j)
        
        # For each diagonal neighbor l, check whether i and l have two common edge neighbors
        for l in diag_candidates:
            if l <= i:  # Avoid duplicates
                continue
            common_edge_neighbors = (
                set(edge_neighbors.get(i, [])) & set(edge_neighbors.get(l, []))
            )
            
            if len(common_edge_neighbors) >= 2:
                # If i and l have at least two common edge neighbors, they are diagonal corners of a square
                # Use the first two common neighbors as the other two square corners
                common_list = sorted(common_edge_neighbors)[:2]
                j, k = common_list
                # The four square corners, sorted and deduplicated
                quad = tuple(sorted([i, j, l, k]))
                quads_set.add(quad)
    
    return list(quads_set)


def densify_st_slice(
    adata,
    coor_key="spatial",
    c2c_dist=None,
    tolerance=0.3,
    add_corner_voids=True,
):
    """
    Densify voids in a single ST slice.
    
    Includes two void types:
    - Edge voids: midpoints between adjacent spot pairs (four per spot)
    - Corner voids: centers of four-spot squares (optional)
    
    Parameters
    ----------
    add_corner_voids : bool
        Whether to also insert corner voids at square centers.
        True: each spot has four edge voids and four corner voids nearby; theoretical densification is about 4x.
        False: add edge voids only; theoretical densification is about 3x.
    """
    coords_full = np.asarray(adata.obsm[coor_key])
    if coords_full.shape[1] >= 2:
        coords = coords_full[:, :2]
    else:
        raise ValueError(f"Insufficient coordinate dimensions: {coords_full.shape}")
    
    if c2c_dist is None:
        tree = cKDTree(coords)
        dists, _ = tree.query(coords, k=2)
        c2c_dist = float(np.median(dists[:, 1]))
        print(f"  Automatically inferred c2c_dist = {c2c_dist:.2f}")
    
    edge_neighbors = find_edge_neighbors_st(coords, c2c_dist, tolerance)
    
    n_neighbors_dist = pd.Series(
        [len(v) for v in edge_neighbors.values()]
    ).value_counts().sort_index()
    print(f"  Neighbor-count distribution: {dict(n_neighbors_dist)}")
    
    # ---------- Edge voids ----------
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
            "void_id": f"edge_{k}",
            "void_type": "edge",
            "src_idx1": i,
            "src_idx2": j,
            "src_idx3": np.nan,
            "src_idx4": np.nan,
            "x": midpoint[0],
            "y": midpoint[1],
        })
    
    n_edge = len(new_coords)
    print(f"  Edge voids: {n_edge}")
    
    # ---------- Corner voids ----------
    if add_corner_voids:
        quads = find_corner_quads_st(coords, edge_neighbors, c2c_dist, tolerance)
        
        for k, quad in enumerate(quads):
            center = np.mean([coords[idx] for idx in quad], axis=0)
            new_coords.append(center)
            void_info.append({
                "void_id": f"corner_{k}",
                "void_type": "corner",
                "src_idx1": quad[0],
                "src_idx2": quad[1],
                "src_idx3": quad[2],
                "src_idx4": quad[3],
                "x": center[0],
                "y": center[1],
            })
        
        n_corner = len(new_coords) - n_edge
        print(f"  Corner voids: {n_corner}")
    
    new_coords = np.array(new_coords)
    void_info_df = pd.DataFrame(void_info).set_index("void_id")
    
    print(f"  Number of original spots: {len(coords)}")
    print(f"  Total inserted voids: {len(new_coords)}")
    print(f"  Densification factor: {(len(coords) + len(new_coords)) / len(coords):.2f}x")
    
    return new_coords, void_info_df


# ============================================================
# Part 2: Build the densified AnnData object; unchanged
# ============================================================

def build_densified_adata(
    adata_original,
    new_coords,
    coor_key="spatial",
    coor_3d_key="3D_coor",
    z_value=None,
    inherit_obs_cols=("slice",),
):
    """Merge newly inserted spots into the original adata object."""
    n_new = len(new_coords)
    
    X_new = np.full((n_new, adata_original.n_vars), np.nan, dtype=np.float32)
    
    obs_new = pd.DataFrame(index=[f"virtual_{i}" for i in range(n_new)])
    for col in adata_original.obs.columns:
        if col in inherit_obs_cols and col in adata_original.obs.columns:
            unique_vals = adata_original.obs[col].unique()
            if len(unique_vals) == 1:
                obs_new[col] = unique_vals[0]
            else:
                print(f"  Warning: Column '{col}' has non-unique values in the original slice; fill this column with NaN for virtual spots")
                obs_new[col] = np.nan
        else:
            obs_new[col] = np.nan
    obs_new["spot_type"] = "virtual"
    
    obs_orig = adata_original.obs.copy()
    obs_orig["spot_type"] = "original"
    
    obsm_new = {}
    
    orig_coor = np.asarray(adata_original.obsm[coor_key])
    if orig_coor.shape[1] == 2:
        obsm_new[coor_key] = np.vstack([orig_coor, new_coords])
    elif orig_coor.shape[1] == 3:
        if z_value is None:
            z_value = float(orig_coor[:, 2].mean())
        new_3d = np.column_stack([new_coords, np.full(n_new, z_value)])
        obsm_new[coor_key] = np.vstack([orig_coor, new_3d])
    
    if coor_3d_key is not None and coor_3d_key != coor_key:
        if coor_3d_key in adata_original.obsm:
            orig_3d = np.asarray(adata_original.obsm[coor_3d_key])
            if z_value is None:
                z_value = float(orig_3d[:, 2].mean())
            new_3d = np.column_stack([new_coords, np.full(n_new, z_value)])
            obsm_new[coor_3d_key] = np.vstack([orig_3d, new_3d])
    
    X_orig = adata_original.X.toarray() if hasattr(adata_original.X, 'toarray') else adata_original.X
    
    adata_dense = ad.AnnData(
        X=np.vstack([X_orig, X_new]),
        obs=pd.concat([obs_orig, obs_new], axis=0),
        var=adata_original.var.copy(),
        obsm=obsm_new,
    )
    
    return adata_dense


# ============================================================
# Part 3: Geometry visualization; updated to distinguish edge voids and corner voids
# ============================================================

def visualize_densification(
    coords_orig,
    coords_new,
    void_info=None,  # Added: pass void_info to distinguish edge voids from corner voids
    spot_radius=None,
    c2c_dist=None,
    save_path=None,
):
    """
    Multi-level visualization of densification effects for the ST square-grid version.
    
    Parameters
    ----------
    void_info : pd.DataFrame, optional
        void_info_df returned by densify_st_slice.
        If provided, different colors distinguish edge voids (red) from corner voids.
    """
    if c2c_dist is None:
        tree = cKDTree(coords_orig)
        dists, _ = tree.query(coords_orig, k=2)
        c2c_dist = float(np.median(dists[:, 1]))
    
    if spot_radius is None:
        spot_radius = c2c_dist * 0.25
    
    # Distinguish edge voids from corner voids
    if void_info is not None and 'void_type' in void_info.columns:
        edge_mask = (void_info['void_type'] == 'edge').values
        corner_mask = (void_info['void_type'] == 'corner').values
        coords_edge = coords_new[edge_mask]
        coords_corner = coords_new[corner_mask]
        has_corner = len(coords_corner) > 0
    else:
        coords_edge = coords_new
        coords_corner = np.empty((0, 2))
        has_corner = False
    
    EDGE_COLOR = '#E63946'    # red
    CORNER_COLOR = '#F77F00'  # orange
    
    fig = plt.figure(figsize=(20, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1.4], wspace=0.25)
    
    # ============================================================
    # Panel A: Global morphology
    # ============================================================
    ax_a = fig.add_subplot(gs[0])
    ax_a.scatter(coords_orig[:, 0], coords_orig[:, 1], 
                 s=4, c='#2E86AB', alpha=0.8, label=f'Original ({len(coords_orig)})')
    ax_a.scatter(coords_edge[:, 0], coords_edge[:, 1],
                 s=2, c=EDGE_COLOR, alpha=0.6, label=f'Edge voids ({len(coords_edge)})')
    if has_corner:
        ax_a.scatter(coords_corner[:, 0], coords_corner[:, 1],
                     s=2, c=CORNER_COLOR, alpha=0.6, label=f'Corner voids ({len(coords_corner)})')
    
    ax_a.set_aspect('equal')
    ax_a.invert_yaxis()
    ax_a.legend(loc='upper right', fontsize=10, framealpha=0.95)
    ax_a.set_title('A. Global tissue coverage', fontsize=13, fontweight='bold')
    ax_a.set_xlabel('x')
    ax_a.set_ylabel('y')
    
    center_idx = len(coords_orig) // 2
    cx, cy = coords_orig[center_idx]
    
    box_b_size = c2c_dist * 1.5
    rect_b = Rectangle((cx - box_b_size, cy - box_b_size),
                        2 * box_b_size, 2 * box_b_size,
                        fill=False, edgecolor='orange', linewidth=2.5)
    ax_a.add_patch(rect_b)
    ax_a.text(cx + box_b_size, cy - box_b_size, 'B', 
              fontsize=14, fontweight='bold', color='orange',
              verticalalignment='bottom')
    
    box_c_size = c2c_dist * 4
    rect_c = Rectangle((cx - box_c_size, cy - box_c_size),
                        2 * box_c_size, 2 * box_c_size,
                        fill=False, edgecolor='green', linewidth=2.5)
    ax_a.add_patch(rect_c)
    ax_a.text(cx + box_c_size, cy - box_c_size, 'C',
              fontsize=14, fontweight='bold', color='green',
              verticalalignment='bottom')
    
   # ============================================================
    # Panel B: Single spot + four neighbors + nearby virtual spots
    # ============================================================
    ax_b = fig.add_subplot(gs[1])
    
    tree = cKDTree(coords_orig)
    dists, idxs = tree.query([cx, cy], k=8)
    neighbor_idxs = []
    for d, i in zip(dists[1:], idxs[1:]):
        if d <= c2c_dist * 1.3:
            neighbor_idxs.append(i)
    neighbor_idxs = neighbor_idxs[:4]
    
    # Find all virtual spots in the Panel B region; draw both edge and corner spots
    panel_b_half = box_b_size  # Panel B half-width
    relevant_voids = []
    for nv in coords_new:
        # Within the rectangular Panel B region
        if (abs(nv[0] - cx) < panel_b_half and abs(nv[1] - cy) < panel_b_half):
            relevant_voids.append(nv)
    relevant_voids = np.array(relevant_voids) if relevant_voids else np.empty((0, 2))
    
    # Center spot
    circle_center = Circle((cx, cy), spot_radius, 
                            facecolor='#1A4F72', edgecolor='black',
                            linewidth=1.5, alpha=0.95, zorder=3)
    ax_b.add_patch(circle_center)
    ax_b.text(cx, cy, 'A', ha='center', va='center', 
              color='white', fontsize=11, fontweight='bold', zorder=4)
    
    # Four neighbors
    for k, i in enumerate(neighbor_idxs):
        nx, ny = coords_orig[i]
        circle = Circle((nx, ny), spot_radius,
                        facecolor='#5DADE2', edgecolor='black',
                        linewidth=1.0, alpha=0.85, zorder=3)
        ax_b.add_patch(circle)
        ax_b.text(nx, ny, f'N{k+1}', ha='center', va='center',
                  color='white', fontsize=9, fontweight='bold', zorder=4)
        ax_b.plot([cx, nx], [cy, ny], '-', color='#888', 
                  alpha=0.5, linewidth=1.0, zorder=1)
    
    # All virtual spots: red dashed circles plus center points
    for vx, vy in relevant_voids:
        circle_v = Circle((vx, vy), spot_radius,
                          facecolor='none', edgecolor=EDGE_COLOR,
                          linewidth=1.8, alpha=0.9, zorder=2,
                          linestyle='--')
        ax_b.add_patch(circle_v)
        ax_b.plot(vx, vy, 'o', color=EDGE_COLOR, markersize=4, zorder=4)
    
    ax_b.set_xlim(cx - box_b_size, cx + box_b_size)
    ax_b.set_ylim(cy - box_b_size, cy + box_b_size)
    ax_b.set_aspect('equal')
    ax_b.invert_yaxis()
    ax_b.set_title('B. Geometry: 1 spot + 4 neighbors + virtual spots',
                   fontsize=13, fontweight='bold')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1A4F72',
               markersize=12, markeredgecolor='black', label='Original spot'),
        Line2D([0], [0], marker='o', color=EDGE_COLOR, markerfacecolor='none',
               markersize=12, markeredgewidth=1.8, linestyle='--',
               label='Virtual spot'),
    ]
    ax_b.legend(handles=legend_elements, loc='upper left', fontsize=9,
                framealpha=0.95)
    
    # ============================================================
    # Panel C: Medium-size region for inspecting grid structure
    # ============================================================
    ax_c = fig.add_subplot(gs[2])
    
    mask_orig = (
        (coords_orig[:, 0] > cx - box_c_size) & (coords_orig[:, 0] < cx + box_c_size) &
        (coords_orig[:, 1] > cy - box_c_size) & (coords_orig[:, 1] < cy + box_c_size)
    )
    mask_edge = (
        (coords_edge[:, 0] > cx - box_c_size) & (coords_edge[:, 0] < cx + box_c_size) &
        (coords_edge[:, 1] > cy - box_c_size) & (coords_edge[:, 1] < cy + box_c_size)
    )
    coords_orig_sub = coords_orig[mask_orig]
    coords_edge_sub = coords_edge[mask_edge]
    
    if has_corner:
        mask_corner = (
            (coords_corner[:, 0] > cx - box_c_size) & (coords_corner[:, 0] < cx + box_c_size) &
            (coords_corner[:, 1] > cy - box_c_size) & (coords_corner[:, 1] < cy + box_c_size)
        )
        coords_corner_sub = coords_corner[mask_corner]
    else:
        coords_corner_sub = np.empty((0, 2))
    
    for x, y in coords_orig_sub:
        circle = Circle((x, y), spot_radius,
                        facecolor='#2E86AB', edgecolor='black',
                        linewidth=0.6, alpha=0.9, zorder=3)
        ax_c.add_patch(circle)
    
    for x, y in coords_edge_sub:
        circle = Circle((x, y), spot_radius,
                        facecolor='none', edgecolor=EDGE_COLOR,
                        linewidth=1.0, alpha=0.85, zorder=2,
                        linestyle='--')
        ax_c.add_patch(circle)
    
    for x, y in coords_corner_sub:
        circle = Circle((x, y), spot_radius,
                        facecolor='none', edgecolor=CORNER_COLOR,
                        linewidth=1.0, alpha=0.85, zorder=2,
                        linestyle=':')
        ax_c.add_patch(circle)
    
    ax_c.set_xlim(cx - box_c_size, cx + box_c_size)
    ax_c.set_ylim(cy - box_c_size, cy + box_c_size)
    ax_c.set_aspect('equal')
    ax_c.invert_yaxis()
    
    title_c = (f'C. Square lattice ({len(coords_orig_sub)} originals + '
               f'{len(coords_edge_sub)} edge')
    if has_corner:
        title_c += f' + {len(coords_corner_sub)} corner)'
    else:
        title_c += ')'
    ax_c.set_title(title_c, fontsize=13, fontweight='bold')
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    
    plt.suptitle(
        f'ST Slice Densification: {len(coords_orig)} -> {len(coords_orig) + len(coords_new)} spots '
        f'({(len(coords_orig) + len(coords_new))/len(coords_orig):.2f}x density)',
        fontsize=15, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()

# ============================================================
# Part 4: Gene-expression visualization
# ============================================================

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
    adata_slice = adata_super[adata_super.obs['slice'] == slice_id].copy()
    
    is_orig = (adata_slice.obs['spot_type'] == 'original').values
    coords = adata_slice.obsm[coor_key][:, :2]
    
    if gene not in adata_slice.var_names:
        print(f"  Warning: {gene} is not in var_names; skipping")
        return
    gene_idx = adata_slice.var_names.get_loc(gene)
    
    X_full = adata_slice.X
    if hasattr(X_full, 'toarray'):
        X_full = X_full.toarray()
    expr_real_orig = X_full[is_orig, gene_idx]
    
    expr_pred_all = adata_slice.obsm['X_pred'][:, gene_idx]
    
    vmin = float(np.percentile(expr_real_orig, 2))
    vmax = float(np.percentile(expr_real_orig, 98))
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
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
    
    sc2 = axes[1].scatter(
        coords[:, 0], coords[:, 1],
        c=expr_pred_all, cmap=cmap, norm=norm,
        s=8, edgecolors='none',
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
    slice_ids=None,
    slice_names=None,
    coor_key='3D_coor',
    cmap='viridis',
    save_path=None,
):
    """
    Compare original versus densified results for one gene across multiple slices using an N x 2 grid.
    """
    if slice_ids is None:
        slice_ids = sorted(adata_st.obs['slice'].unique())
    if slice_names is None:
        slice_names = [f'Slice {sid}' for sid in slice_ids]
    
    if gene not in adata_super.var_names:
        print(f"  Warning: {gene} is not in var_names; skipping")
        return
    
    gene_idx = adata_super.var_names.get_loc(gene)
    
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
        adata_orig_i = adata_st[adata_st.obs['slice'] == sid]
        coords_orig = adata_orig_i.obsm[coor_key][:, :2]
        X_i = adata_orig_i.X
        if hasattr(X_i, 'toarray'):
            X_i = X_i.toarray()
        expr_orig = X_i[:, gene_idx]
        
        adata_super_i = adata_super[adata_super.obs['slice'] == sid]
        coords_super = adata_super_i.obsm[coor_key][:, :2]
        expr_super = adata_super_i.obsm['X_pred'][:, gene_idx]
        
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
        
        ax_r = axes[row, 1]
        sc_r = ax_r.scatter(
            coords_super[:, 0], coords_super[:, 1],
            c=expr_super, cmap=cmap, norm=norm, s=8, edgecolors='none',
        )
        ax_r.set_aspect('equal')
        ax_r.invert_yaxis()
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        if row == 0:
            ax_r.set_title('Super-resolution', fontsize=13, fontweight='bold')
    
    fig.colorbar(sc_r, ax=axes, shrink=0.5, label=gene, location='right', pad=0.02)
    
    plt.suptitle(f'{gene} across {len(slice_ids)} slices',
                 fontsize=15, fontweight='bold', y=1.005)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Save: {save_path}")
    plt.show()