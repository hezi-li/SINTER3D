import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
import os
from SINTER3D.networks_spatial import DeconvNet
from SINTER3D.utils import set_seed
import json


def load_config(config_name_or_path):
    """加载配置文件，支持单文件或多配置"""
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
                print(f"✅ Loaded configuration '{config_key}' from {config_path}")
                return config_data[config_key]
            else:
                print(f"❌ Error: Config '{config_key}' not found. Available: {list(config_data.keys())}")
                return {}
        
        if isinstance(config_data, dict) and all(isinstance(v, dict) for v in config_data.values()):
            available_keys = list(config_data.keys())
            print(f"⚠️ Multiple configs found: {available_keys}")
            print(f"Using '{available_keys[0]}' as default")
            return config_data[available_keys[0]]
        else:
            return config_data
            
    except FileNotFoundError:
        print(f"⚠️ Warning: Config file '{config_path}' not found, using default parameters")
        return {}


class Model():
    def __init__(self, adata_st_list_raw, adata_st, adata_basis, slice_idx,
                 config=None, 
                 hidden_dims=[512, 128],
                 slice_emb_dim=16,
                 training_steps=11,
                 lr=0.001,
                 seed=2025,
                 patience=200,
                 mid_channel=200,
                 save_path='./results_DLPFC',
                 use_type='train',
                 normalize=100,
                 coord_emb_dim=64,   
                 num_freqs=6,         
                 alpha_poisson = 5,
                 lambda_feature = 1,
                 gamma_feature = 2
                 ):

        # 1️⃣ 收集默认参数
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
            coord_emb_dim=coord_emb_dim,
            num_freqs=num_freqs,
            alpha_poisson = alpha_poisson,
            lambda_feature = lambda_feature,
            gamma_feature = gamma_feature
        )

        # 2️⃣ 读取 config 覆盖默认值
        if config is not None:
            config_params = load_config(config)
            print("📄 Config parameters loaded:", config_params)
            params.update(config_params)  # config 优先级高于默认值

        # 3️⃣ 全部赋值到 self
        for k, v in params.items():
            setattr(self, k, v)

        # 4️⃣ 打印最终生效参数（调试用）
        print("\n✅ Final parameters used in Model:")
        for k in params.keys():
            print(f"  {k}: {getattr(self, k)}")
        print()

        # ---- 接下来固定初始化部分 ----
        set_seed(self.seed)
        self.adata_basis = adata_basis
        self.adata_st = adata_st
        self.celltypes = list(adata_basis.obs.index)
        self.adata_st_list_raw = adata_st_list_raw
        self.slice_idx = slice_idx

        # slice 编码
        unique_slices = sorted(self.adata_st.obs["slice"].unique())
        self.slice_remap = {old: i for i, old in enumerate(unique_slices)}
        self.adata_st.obs["slice"] = self.adata_st.obs["slice"].map(self.slice_remap).astype(int)

        # 设备
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # 确认 hidden_dims 格式
        self.hidden_dims = [adata_st.shape[1]] + self.hidden_dims
        self.n_celltype = adata_basis.shape[0]
        self.n_slices = len(unique_slices)

        # 构建网络
        self.net = DeconvNet(
            hidden_dims=self.hidden_dims,
            n_celltypes=self.n_celltype,
            n_slices=self.n_slices,
            slice_emb_dim=self.slice_emb_dim,
            training_steps=self.training_steps,
            mid_channel=self.mid_channel,
            coord_emb_dim=self.coord_emb_dim,
            num_freqs=self.num_freqs,
            c = normalize,
            alpha_poisson = self.alpha_poisson,
            lambda_feature = self.lambda_feature,
            gamma_feature = self.gamma_feature
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=self.lr)

        # 读取数据
        if scipy.sparse.issparse(adata_st.X):
            self.X = torch.from_numpy(adata_st.X.toarray()).float().to(self.device)
        else:
            self.X = torch.from_numpy(adata_st.X).float().to(self.device)

        self.Y = torch.from_numpy(np.array(adata_st.obsm["count"])).float().to(self.device)
        self.lY = torch.from_numpy(np.array(adata_st.obs["library_size"].values.reshape(-1, 1))).float().to(self.device)
        self.slice = torch.from_numpy(np.array(adata_st.obs["slice"].values)).long().to(self.device)
        self.basis = torch.from_numpy(np.array(adata_basis.X)).float().to(self.device)
        self.coord = torch.from_numpy(np.array(adata_st.obsm['3D_coor'])).float().to(self.device)

        # --------- train with early stopping ---------
    def train(self, report_loss=True, step_interval=1000, min_delta=1e-4):
        """
        训练模型，支持早停
        min_delta: loss 改善的最小幅度，小于这个认为没有进步
        """
        self.net.train()

        best_loss = float('inf')  # 当前最佳loss
        best_state = None  # 最佳模型参数
        wait_count = 0  # 等待计数器（耐心值累积）

        for step in tqdm(range(self.net.training_steps)):
            # 前向计算
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

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 打印日志
            if report_loss and step % step_interval == 0:
                print(f"Step {step} | Loss: {loss.item():.6f} | Best: {best_loss:.6f}")

            # ========== 早停判断 ==========
            if best_loss - loss.item() > min_delta:
                best_loss = loss.item()
                best_state = {
                    'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                wait_count = 0  # 重置等待计数器
            else:
                wait_count += 1

            # 如果等待次数超过耐心值 => 提前停止
            if wait_count >= self.patience:
                print(f"⏹ Early stopping at step {step} | Best loss: {best_loss:.6f}")
                break

        # 训练结束后加载最佳模型
        if best_state is not None:
            self.net.load_state_dict(best_state['model'])
            self.optimizer.load_state_dict(best_state['optimizer'])
            print("✅ Loaded best model state from training.")


    def eval(self, adata_st_list_raw, save=False, output_path="./results"):
        self.net.eval()
        self.Z, self.beta, self.alpha, self.gamma = self.net.evaluate(
            self.coord, self.X, self.slice
        )

        # latent embedding
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names, ].values

        self.latent = cell_reps.loc[self.adata_st.obs_names, ].values
        if save:
            cell_reps.to_csv(os.path.join(output_path, "representation.csv"))

        # 反卷积结果
        b = self.beta.detach().cpu().numpy()
        n_spots = 0
        adata_st_decon_list = []
        decon_results_all = []  # 用于保存所有的反卷积结果

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

            # 生成反卷积结果
            decon_res = pd.DataFrame(
                b[n_spots:(n_spots + adata_st_i.shape[0]), :],
                columns=self.celltypes
            )
            decon_res.index = adata_st_i.obs.index
            adata_st_i_obs = adata_st_i.obs.drop(columns=self.celltypes, errors="ignore")
            adata_st_i.obs = adata_st_i_obs.join(decon_res)

            # 将反卷积结果添加到列表中
            decon_results_all.append(decon_res)

            n_spots += adata_st_i.shape[0]
            adata_st_decon_list.append(adata_st_i)

        # 如果需要保存反卷积结果，将所有反卷积结果合并后保存
        if save:
            # 将反卷积结果合并为一个 DataFrame
            decon_results_combined = pd.concat(decon_results_all)
            decon_results_combined.to_csv(os.path.join(output_path, "deconvolution_results.csv"))

        return adata_st_decon_list

    # -------- inference_latent ---------
    def inference_latent(self, adata_new, coord_key="3D_coor", method="distance", 
                        decode=False, deconv=False):
        self.net.eval()
        coord_new = torch.from_numpy(np.array(adata_new.obsm[coord_key])).float().to(self.device)

        with torch.no_grad():
            # 用模型的切片数量来保证和 slice_emb 权重长度一致
            n_slices_model = self.net.slice_emb.weight.data.shape[0]
            centers = []
            for s in range(n_slices_model):
                coords_s = self.adata_st[self.adata_st.obs["slice"] == s].obsm[coord_key]
                centers.append(coords_s.mean(0))
            centers = np.array(centers)

            coords_new_center = np.array(adata_new.obsm[coord_key]).mean(0)
            dists = np.linalg.norm(centers - coords_new_center, axis=1)
            weights = np.exp(-dists / (dists.std() + 1e-8))
            weights = weights / weights.sum()

            slice_embs = self.net.slice_emb.weight.data.cpu().numpy()
            new_slice_emb = (weights[:, None] * slice_embs).sum(0, keepdims=True)
            new_slice_emb = torch.from_numpy(new_slice_emb).float().to(self.device)

            # 临时扩展 slice embedding 表
            original_weight = self.net.slice_emb.weight.data.clone()
            self.net.slice_emb.weight = torch.nn.Parameter(
                torch.cat([original_weight, new_slice_emb], dim=0)
            )
            slice_new_id = self.n_slices
            slice_new = torch.full((adata_new.shape[0],), slice_new_id, dtype=torch.long).to(self.device)

            # 编码
            Z_new, _ = self.net.inference_encoder(coord_new)
            Z_new = Z_new.float()

            if decode:
                X_pred = self.net.decoder(Z_new)
                adata_new.obsm["X_pred"] = X_pred.cpu().numpy()

            if deconv:
                slice_new_emb = self.net.slice_emb(slice_new).float()
                beta_new, alpha_new = self.net.deconvolutioner(Z_new, slice_new_emb)
                cell_type_props = beta_new.cpu().numpy()

                prop_df = pd.DataFrame(cell_type_props, columns=self.celltypes, index=adata_new.obs.index)
                prop_df.to_csv("cell_type_proportions.csv")
                adata_new.obs = adata_new.obs.drop(columns=self.celltypes, errors="ignore")
                for cell_type in self.celltypes:
                    adata_new.obs[cell_type] = prop_df[cell_type]
               
                adata_new.obsm["cell_type_proportions"] = cell_type_props
                adata_new.obsm["deconv_alpha"] = alpha_new.cpu().numpy()

            # 恢复原始 embedding 表
            self.net.slice_emb.weight = torch.nn.Parameter(original_weight)

            adata_new.obsm["latent"] = Z_new.cpu().numpy()

        return adata_new

    