import joblib
import numpy as np
import scanpy as sc

from SINTER3D.model_basic import Model as BaseModel
from SINTER3D.model_spatial import Model as MultiScaleModel
from SINTER3D.model_complex import Model as complexModel

# ========= 特征提取函数（必须和训练选择器时一致） =========
def extract_dataset_features(adata_st):
    if "slice" not in adata_st.obs.columns:
        raise ValueError("AnnData.obs 中没有 'slice' 列，无法计算切片数")
    n_slices = adata_st.obs["slice"].nunique()
    X_dense = adata_st.X.toarray() if hasattr(adata_st.X, 'toarray') else adata_st.X
    expr_complexity = float(np.mean(np.var(X_dense, axis=0)))
    return [n_slices, expr_complexity]

# ========= 自动选择并训练 =========
def auto_select_and_train(
    adata_st_list_raw,
    adata_st,
    adata_basis,
    slice_idx,
    selector_path="model_selector.pkl",
    config=None
):
    # 1. 加载选择器
    clf = joblib.load(selector_path)

    # 2. 提取特征
    features = [extract_dataset_features(adata_st)]
    n_slices, expr_complexity = features[0]
    print(f"[特征] n_slices={n_slices}, expr_complexity={expr_complexity:.6f}")

    # 3. 预测最佳模型
    model_type = clf.predict(features)[0]
    print(f" 数据驱动选择模型: {model_type.upper()}")

    # 4. 初始化对应模型
    if model_type == "basic":
        model = BaseModel(adata_st_list_raw, adata_st, adata_basis, slice_idx, config=config)
    elif model_type == "spatial":
        model = MultiScaleModel(adata_st_list_raw, adata_st, adata_basis, slice_idx, config=config)
    elif model_type == "complex":
        model = complexModel(adata_st_list_raw, adata_st, adata_basis, slice_idx, config=config)
    else:
        raise ValueError(f"未知模型类别: {model_type}")

    # 5. 开始训练
    model.train()

    return model

