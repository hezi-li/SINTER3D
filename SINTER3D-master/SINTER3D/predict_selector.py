import joblib
import numpy as np
import scanpy as sc

from SINTER3D.model_basic import Model as BaseModel
from SINTER3D.model_spatial import Model as MultiScaleModel
from SINTER3D.model_complex import Model as complexModel

# ========= Feature extraction utilities; must match selector training. =========
def extract_dataset_features(adata_st):
    if "slice" not in adata_st.obs.columns:
        raise ValueError("AnnData.obs does not contain a 'slice' column; cannot compute the number of slices")
    n_slices = adata_st.obs["slice"].nunique()
    X_dense = adata_st.X.toarray() if hasattr(adata_st.X, 'toarray') else adata_st.X
    expr_complexity = float(np.mean(np.var(X_dense, axis=0)))
    return [n_slices, expr_complexity]

# ========= Automatically select and train =========
def auto_select_and_train(
    adata_st_list_raw,
    adata_st,
    adata_basis,
    slice_idx,
    selector_path="model_selector.pkl",
    config=None
):
    # 1. Load the selector
    clf = joblib.load(selector_path)

    # 2. Extract features
    features = [extract_dataset_features(adata_st)]
    n_slices, expr_complexity = features[0]
    print(f"[Features] n_slices={n_slices}, expr_complexity={expr_complexity:.6f}")

    # 3. Predict the best model
    model_type = clf.predict(features)[0]
    print(f" Data-driven selected model: {model_type.upper()}")

    # 4. Initialize the corresponding model
    if model_type == "basic":
        model = BaseModel(adata_st_list_raw, adata_st, adata_basis, slice_idx, config=config)
    elif model_type == "spatial":
        model = MultiScaleModel(adata_st_list_raw, adata_st, adata_basis, slice_idx, config=config)
    elif model_type == "complex":
        model = complexModel(adata_st_list_raw, adata_st, adata_basis, slice_idx, config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 5. Start training
    model.train()

    return model

