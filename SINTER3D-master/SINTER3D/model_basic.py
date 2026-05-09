import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
import os
from SINTER3D.networks_basic import DeconvNet
from SINTER3D.utils import set_seed
import json


def load_config(config_name_or_path):
    """Load configuration files; supports single-file or multi-configuration formats."""
    if config_name_or_path is None:
        return {}
    
    config_key = None
    if ':' in str(config_name_or_path):
        file_part, config_key = str(config_name_or_path).split(':', 1)
        config_name_or_path = file_part
    
    if config_name_or_path.endswith('.json'):
        config_path = config_name_or_path
    else:
        config_path = f"./configs/{config_name_or_path}.json"
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        if config_key is not None:
            if config_key in config_data:
                print(f"Loaded configuration '{config_key}' from {config_path}")
                return config_data[config_key]
            else:
                print(f"Config '{config_key}' not found. Available: {list(config_data.keys())}")
                return {}
        
        if isinstance(config_data, dict) and all(isinstance(v, dict) for v in config_data.values()):
            available_keys = list(config_data.keys())
            print(f"Warning: Multiple configs found: {available_keys}")
            print(f"Using '{available_keys[0]}' as default")
            return config_data[available_keys[0]]
        else:
            return config_data
            
    except FileNotFoundError:
        print(f"Warning: Config file '{config_path}' not found, using default parameters")
        return {}


class Model():
    def __init__(self, adata_st_list_raw, adata_st, adata_basis, slice_idx,
                 config=None, 
                 hidden_dims=[512, 128],
                 slice_emb_dim=16,
                 training_steps=10000,
                 lr=0.001,
                 seed=2025,
                 patience=200,
                 mid_channel=200,
                 save_path='./results_DLPFC',
                 use_type='train',
                 normalize=100,
                 alpha_poisson = 5,
                 lambda_feature = 1,
                 gamma_feature = 2
                 ):

        # 1. Collect default parameters
        params = dict(
            hidden_dims=hidden_dims,
            slice_emb_dim=slice_emb_dim,
            training_steps=training_steps,
            lr=lr,
            seed=seed,
            patience=patience,
            mid_channel=mid_channel,
            save_path=save_path,
            use_type=use_type,
            normalize=normalize,
            alpha_poisson = alpha_poisson,
            lambda_feature = lambda_feature,
            gamma_feature = gamma_feature,
        )

        # 2. Override default parameters with config values
        if config is not None:
            config_params = load_config(config)
            print("Config parameters loaded:", config_params)
            params.update(config_params)  # Config values have the highest priority

        # 3. Assign values to self
        for k, v in params.items():
            setattr(self, k, v)

        # 4. Debug final parameters
        print("\nFinal parameters in Model:")
        for k, v in params.items():
            print(f"  {k}: {v}")
        print()

        # Fixed values
        set_seed(self.seed)
        self.adata_basis = adata_basis
        self.adata_st = adata_st
        self.celltypes = list(adata_basis.obs.index)
        self.adata_st_list_raw = adata_st_list_raw
        self.slice_idx = slice_idx

        unique_slices = sorted(self.adata_st.obs["slice"].unique())
        self.slice_remap = {old: i for i, old in enumerate(unique_slices)}
        self.adata_st.obs["slice"] = self.adata_st.obs["slice"].map(self.slice_remap).astype(int)

        # Device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Network
        self.hidden_dims = [adata_st.shape[1]] + self.hidden_dims
        self.n_celltype = adata_basis.shape[0]
        self.n_slices = len(unique_slices)

        self.net = DeconvNet(
            hidden_dims=self.hidden_dims, 
            n_celltypes=self.n_celltype, 
            n_slices=self.n_slices, 
            slice_emb_dim=self.slice_emb_dim, 
            training_steps=self.training_steps, 
            mid_channel=self.mid_channel,
            alpha_poisson = self.alpha_poisson,
            lambda_feature = self.lambda_feature,
            gamma_feature = self.gamma_feature
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=self.lr)

        # Data
        if scipy.sparse.issparse(adata_st.X):
            self.X = torch.from_numpy(adata_st.X.toarray()).float().to(self.device)
        else:
            self.X = torch.from_numpy(adata_st.X).float().to(self.device)
        
        self.Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.slice = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(self.device)
        self.basis = torch.from_numpy(np.array(adata_basis.X)).float().to(self.device)
        self.coord = torch.from_numpy(np.array(adata_st.obsm['3D_coor'])).float().to(self.device)

    def train(self, report_loss=True, step_interval=1000, min_delta=1e-4):
        """
        Train the model with early stopping support.
        min_delta: Minimum loss improvement; smaller changes are treated as no improvement.
        """
        self.net.train()

        # Track the best loss and corresponding model state
        best_loss = float('inf')
        best_state = None
        wait_count = 0  # Patience counter

        for step in tqdm(range(self.net.training_steps)):
            # Forward pass
            loss, recon, denoise, Z_, ind_min, ind_max = self.net(
                coord=self.coord,
                node_feats=self.X, 
                count_matrix=self.Y, 
                library_size=self.lY, 
                slice_label=self.slice, 
                basis=self.basis,
                c=self.normalize,
                step=step
            )

            # Backward propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log output
            if report_loss and step % step_interval == 0:
                print(f"[Step {step}] Loss={loss.item():.6f} | Best={best_loss:.6f}")

            # -------- Early stopping logic --------
            if best_loss - loss.item() > min_delta:

                best_loss = loss.item()
                best_state = {
                    "model": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict()
                }
                wait_count = 0
            else:
                # Loss did not decrease significantly
                wait_count += 1

            # Trigger early stopping when patience is exhausted
            if wait_count >= self.patience:
                print(f"Early stopping triggered at step {step} | Best Loss={best_loss:.6f}")
                break

        # Restore the best model
        if best_state is not None:
            self.net.load_state_dict(best_state["model"])
            self.optimizer.load_state_dict(best_state["optimizer"])
            print("Loaded best model state from training.")

    def eval(self, adata_st_list_raw, save=False, output_path="./results"):
        self.net.eval()
        self.Z, self.beta, self.alpha, self.gamma = self.net.evaluate(
            self.coord, self.X, self.slice
        )

        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names, ].values
        self.latent = cell_reps.loc[self.adata_st.obs_names, ].values

        b = self.beta.detach().cpu().numpy()
        n_spots = 0
        adata_st_decon_list = []
        decon_results_all = []

        for i, adata_st_i in enumerate(adata_st_list_raw):
            adata_st_i = adata_st_i.copy()
            if self.use_type == 'evaluate':
                if not all(idx.endswith(f"-slice{slice_val}")
                        for idx, slice_val in zip(adata_st_i.obs.index, adata_st_i.obs['slice'])):
                    adata_st_i.obs.index = [
                        f"{idx}-slice{slice_val}" for idx, slice_val in zip(
                            adata_st_i.obs.index, adata_st_i.obs['slice']
                        )
                    ]
            else:
                if not all(idx.endswith(f"-slice{i}") for idx in adata_st_i.obs.index):
                    adata_st_i.obs.index = [f"{idx}-slice{i}" for idx in adata_st_i.obs.index]

            decon_res = pd.DataFrame(
                b[n_spots:(n_spots + adata_st_i.shape[0]), :],
                columns=self.celltypes
            )
            decon_res.index = adata_st_i.obs.index
            decon_results_all.append(decon_res) 

            adata_st_i_obs = adata_st_i.obs.drop(columns=self.celltypes, errors="ignore")
            adata_st_i.obs = adata_st_i_obs.join(decon_res)
            n_spots += adata_st_i.shape[0]
            adata_st_decon_list.append(adata_st_i)

        if save:
            os.makedirs(output_path, exist_ok=True)

            cell_reps.to_csv(os.path.join(output_path, "representation.csv"))
 
            decon_results_combined = pd.concat(decon_results_all)
            decon_results_combined.to_csv(os.path.join(output_path, "deconvolution_results.csv"))

        return adata_st_decon_list

    def inference_latent(
        self,
        adata_new,
        coord_key="3D_coor",
        slice_key="slice",
        method="distance",
        decode=False,
        deconv=False,
    ):
        """
        Inference for one or multiple unseen / virtual sections.
        Each new slice receives an independently interpolated section embedding.
        """
        self.net.eval()

        coord_np = np.array(adata_new.obsm[coord_key])
        coord_new = torch.from_numpy(coord_np).float().to(self.device)

        with torch.no_grad():
            slice_ids = sorted(set(self.adata_st.obs["slice"]))

            centers = []
            valid_slice_ids = []

            for s in slice_ids:
                idx_s = self.adata_st.obs["slice"] == s
                if np.sum(idx_s) == 0:
                    continue
                coords_s = self.adata_st[idx_s].obsm[coord_key]
                centers.append(coords_s.mean(0))
                valid_slice_ids.append(int(s))

            centers = np.array(centers)
            valid_slice_ids = np.array(valid_slice_ids)

            original_weight = self.net.slice_emb.weight.data.clone()
            original_weight_np = original_weight.cpu().numpy()
            n_slices_model = original_weight_np.shape[0]

            if slice_key not in adata_new.obs.columns:
                new_slice_labels = np.array(["new_slice"] * adata_new.shape[0])
            else:
                new_slice_labels = adata_new.obs[slice_key].astype(str).to_numpy()

            unique_new_slices = pd.unique(new_slice_labels)

            new_emb_list = []
            new_slice_id_map = {}

            for k, new_slice in enumerate(unique_new_slices):
                idx_new = new_slice_labels == new_slice
                coords_new_center = coord_np[idx_new].mean(0)

                dists = np.linalg.norm(centers - coords_new_center, axis=1)

                if method == "distance":
                    scale = dists.std() + 1e-8
                    weights = np.exp(-dists / scale)
                    weights = weights / weights.sum()
                elif method == "nearest":
                    weights = np.zeros_like(dists)
                    weights[np.argmin(dists)] = 1.0
                else:
                    raise ValueError("method must be 'distance' or 'nearest'")

                slice_embs = original_weight_np[valid_slice_ids]
                new_slice_emb = (weights[:, None] * slice_embs).sum(0, keepdims=True)

                new_emb_list.append(new_slice_emb)
                new_slice_id_map[new_slice] = n_slices_model + k

            new_embs = np.vstack(new_emb_list)
            new_embs = torch.from_numpy(new_embs).float().to(self.device)

            original_weight = original_weight.float()

            self.net.slice_emb.weight = torch.nn.Parameter(
                torch.cat([original_weight, new_embs], dim=0)
            )

            slice_new_ids = np.array(
                [new_slice_id_map[x] for x in new_slice_labels],
                dtype=np.int64,
            )
            slice_new = torch.from_numpy(slice_new_ids).long().to(self.device)

            Z_new, _ = self.net.inference_encoder(coord_new)

            if decode:
                X_pred = self.net.decoder(Z_new)
                adata_new.obsm["X_pred"] = X_pred.cpu().numpy()

            if deconv:
                slice_new_emb = self.net.slice_emb(slice_new).float()
                beta_new, alpha_new = self.net.deconvolutioner(Z_new, slice_new_emb)
                cell_type_props = beta_new.cpu().numpy()

                prop_df = pd.DataFrame(
                    cell_type_props,
                    columns=self.celltypes,
                    index=adata_new.obs.index,
                )

                adata_new.obs = adata_new.obs.drop(columns=self.celltypes, errors="ignore")
                for cell_type in self.celltypes:
                    adata_new.obs[cell_type] = prop_df[cell_type]

                adata_new.obsm["cell_type_proportions"] = cell_type_props
                adata_new.obsm["deconv_alpha"] = alpha_new.cpu().numpy()

            adata_new.obsm["latent"] = Z_new.cpu().numpy()
            adata_new.obs["_inference_slice_id"] = slice_new_ids.astype(str)
            adata_new.uns["new_slice_id_map"] = {
                str(k): int(v) for k, v in new_slice_id_map.items()
            }

            self.net.slice_emb.weight = torch.nn.Parameter(original_weight)

        return adata_new