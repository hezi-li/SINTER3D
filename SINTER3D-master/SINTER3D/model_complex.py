import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.sparse
from tqdm import tqdm
import os
from SINTER3D.networks_complex import DeconvNet
from SINTER3D.utils import set_seed
import json


def load_config(config_name_or_path):
    """简单的配置加载函数 - 自动处理多配置文件"""
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
        
        # 如果这是多配置文件
        if isinstance(config_data, dict) and all(isinstance(v, dict) for v in config_data.values()):
            available_keys = list(config_data.keys())
            print(f"⚠️ Multiple configs found: {available_keys}")
            print("Please specify config using 'filename.json:config_name' format")
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
                alpha_poisson = 5,
                lambda_feature = 1,
                gamma_feature = 2
                ):

        # 1️⃣ 默认参数收集
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
            gamma_feature = gamma_feature
        )

        # 2️⃣ 加载并覆盖默认值
        if config is not None:
            config_params = load_config(config)
            print("📄 Config parameters loaded:", config_params)
            params.update(config_params) 

        # 3️⃣ 赋值到类属性
        for k, v in params.items():
            setattr(self, k, v)

        # 4️⃣ 启动时打印最终生效参数
        print("\n✅ Final parameters used in Model:")
        for k in params.keys():
            print(f"  {k}: {getattr(self, k)}")
        print()

        # 固定部分
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

        # 这里用 self.hidden_dims（已被 config 覆盖）
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
            alpha_poisson = self.alpha_poisson,
            lambda_feature = self.lambda_feature,
            gamma_feature = self.gamma_feature
        ).to(self.device)

        # 优化器
        self.optimizer = optim.Adamax(list(self.net.parameters()), lr=self.lr)

        # 数据加载
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
        训练模型，支持早停机制
        min_delta: Loss 改善的最小幅度，小于这个值认为没有进步
        """
        self.net.train()

        best_loss = float('inf')     # 当前最佳loss
        best_state = None            # 保存最佳模型状态
        wait_count = 0               # 等待计数器

        for step in tqdm(range(self.net.training_steps)):
            # 前向计算
            loss, Z_teacher, denoise, coord_loss = self.net(
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

            # 日志
            if report_loss and step % step_interval == 0:
                print(f"[Step {step}] Loss={loss.item():.6f} | CoordLoss={coord_loss.item():.6f} | Best={best_loss:.6f}")

            # --------- 早停逻辑 -----------
            if best_loss - loss.item() > min_delta:
                # Loss 有显著改善，更新最佳模型信息
                best_loss = loss.item()
                best_state = {
                    'model': self.net.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }
                wait_count = 0
            else:
                # 没有改善
                wait_count += 1

            # 检查是否超过 patience
            if wait_count >= self.patience:
                print(f"⏹ Early stopping triggered at step {step} | Best Loss={best_loss:.6f}")
                break

        # 恢复最佳模型
        if best_state is not None:
            self.net.load_state_dict(best_state['model'])
            self.optimizer.load_state_dict(best_state['optimizer'])
            print("✅ Loaded best model state from training.")



    def inference_latent(self, adata_new, coord_key="3D_coor", coord_only=False, decode=False):
        self.net.eval()
        coord_new = torch.from_numpy(np.array(adata_new.obsm[coord_key])).float().to(self.device)

        with torch.no_grad():
            Z_new, _ = self.net.inference_encoder(coord_new, coord_only=coord_only)
            if decode and not coord_only:
                X_pred = self.net.decoder(Z_new)
                adata_new.obsm["X_pred"] = X_pred.cpu().numpy()

        adata_new.obsm["latent"] = Z_new.cpu().numpy()
        return adata_new
    

    
    def eval(self, adata_st_list_raw, save=False, output_path="./results"):
        self.net.eval()
        self.Z, self.beta, self.alpha, self.gamma = self.net.evaluate(
            self.coord, self.X, self.slice
        )

        # 处理并保存潜在嵌入（representation.csv）
        embeddings = self.Z.detach().cpu().numpy()
        cell_reps = pd.DataFrame(embeddings)
        cell_reps.index = self.adata_st.obs.index
        self.adata_st.obsm['latent'] = cell_reps.loc[self.adata_st.obs_names, ].values
        self.latent = cell_reps.loc[self.adata_st.obs_names, ].values

        # 处理反卷积结果
        b = self.beta.detach().cpu().numpy()
        n_spots = 0
        adata_st_decon_list = []
        decon_results_all = []  # 新增：用于收集所有切片的反卷积结果

        for i, adata_st_i in enumerate(adata_st_list_raw):
            adata_st_i = adata_st_i.copy()

            # 索引处理（保持原有逻辑）
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

            # 生成当前切片的反卷积结果
            decon_res = pd.DataFrame(
                b[n_spots:(n_spots + adata_st_i.shape[0]), :],
                columns=self.celltypes
            )
            decon_res.index = adata_st_i.obs.index
            decon_results_all.append(decon_res)  # 新增：收集当前切片结果

            # 合并到adata（保持原有逻辑）
            adata_st_i_obs = adata_st_i.obs.drop(columns=self.celltypes, errors="ignore")
            adata_st_i.obs = adata_st_i_obs.join(decon_res)

            n_spots += adata_st_i.shape[0]
            adata_st_decon_list.append(adata_st_i)

        # 新增：保存文件（当save=True时）
        if save:

            # 确保输出目录存在（避免路径不存在报错）
            os.makedirs(output_path, exist_ok=True)
            # 保存潜在嵌入
            cell_reps.to_csv(os.path.join(output_path, "representation.csv"))
            # 合并所有切片的反卷积结果并保存
            decon_results_combined = pd.concat(decon_results_all)
            decon_results_combined.to_csv(os.path.join(output_path, "deconvolution_results.csv"))


        return adata_st_decon_list

