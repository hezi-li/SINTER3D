import pandas as pd
import numpy as np
import anndata as ad
from scipy.spatial import cKDTree
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns

def generate_virtual_slice_from_adata(adata, target_z_list,
                                           boundary_ratio=0.7, lambda_max=1.0, interp_mode='linear',
                                           spatial_name='3D_coor'):
    if not isinstance(target_z_list, (list, np.ndarray)):
        target_z_list = [target_z_list]
    
    # 获取3D坐标
    coords_3d = adata.obsm[spatial_name]
    
    # 从slice列获取切片信息
    slice_labels = adata.obs["slice"].to_numpy()
    unique_slices = np.unique(slice_labels)
    
    # 计算每个切片的平均z坐标
    z_real = []
    slice_to_idx_mapping = {}
    
    for i, slice_label in enumerate(unique_slices):
        slice_mask = slice_labels == slice_label
        slice_coords = coords_3d[slice_mask]
        # 计算该切片的平均z坐标
        avg_z = np.mean(slice_coords[:, 2])
        z_real.append(avg_z)
        slice_to_idx_mapping[slice_label] = i
    
    z_real = np.array(z_real)
    
    # 创建切片索引数组（将slice标签映射为数字索引）
    slice_indices = np.array([slice_to_idx_mapping[label] for label in slice_labels])
    
    virtual_slices_all = []
    
    for target_z in target_z_list:
        # 找到最近的真实切片
        nearest_idx = np.argmin(np.abs(z_real - target_z))
        copied_points = coords_3d[slice_indices == nearest_idx, :2]
        
        # 边界强度
        center = copied_points.mean(axis=0)
        radii = np.linalg.norm(copied_points - center, axis=1)
        max_radius = radii.max()
        boundary_strength = np.clip((radii / max_radius - boundary_ratio) / (1 - boundary_ratio), 0, 1)
        
        # 找到用于插值的上下切片
        if target_z < z_real[nearest_idx]:
            lower_idx = nearest_idx - 1
            upper_idx = nearest_idx
        else:
            lower_idx = nearest_idx
            upper_idx = nearest_idx + 1
        
        lower_idx = np.clip(lower_idx, 0, len(z_real) - 1)
        upper_idx = np.clip(upper_idx, 0, len(z_real) - 1)
        
        # 如果两层是同一层，直接复制
        if lower_idx == upper_idx:
            interp_points = copied_points
        else:
            z0, z1 = z_real[lower_idx], z_real[upper_idx]
            t = (target_z - z0) / (z1 - z0)
            
            # 用 KDTree 在 upper/lower 找最近邻
            lower_points = coords_3d[slice_indices == lower_idx, :2]
            upper_points = coords_3d[slice_indices == upper_idx, :2]
            lower_tree = cKDTree(lower_points)
            upper_tree = cKDTree(upper_points)
            
            # 对 copied_points 找邻居
            _, idx_lower = lower_tree.query(copied_points)
            _, idx_upper = upper_tree.query(copied_points)
            
            interp_points = (1 - t) * lower_points[idx_lower] + t * upper_points[idx_upper]
        
        # 边界融合
        lambda_vals = (lambda_max * boundary_strength)[:, None]
        final_xy = (1 - lambda_vals) * copied_points + lambda_vals * interp_points
        
        # 固定 Z
        final_xyz = np.hstack([final_xy, np.full((final_xy.shape[0], 1), target_z)])
        virtual_slices_all.append(final_xyz)
    
    return np.vstack(virtual_slices_all)

