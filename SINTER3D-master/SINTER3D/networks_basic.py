import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math


class DenseLayer(nn.Module):

    def __init__(self, 
                 c_in, 
                 c_out, 
                 zero_init=False, 
                 ):

        super().__init__()

        self.linear = nn.Linear(c_in, c_out)

        # Initialization
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, 
    			node_feats, # input node features
    			):

        node_feats = self.linear(node_feats)

        return node_feats

class SineLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True, zero_init=False,
                 omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.zero_init = zero_init
        self.in_features = c_in
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        
        if self.zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
            nn.init.zeros_(self.linear.bias.data)
            
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    

class DeconvNet(nn.Module):

    def __init__(self, 
                 hidden_dims,
                 n_celltypes, 
                 n_slices,
                 slice_emb_dim, 
                 training_steps,
                 eps=1e-6,
                 mid_channel=200,
                 alpha_poisson = 5,
                 lambda_feature = 1,
                 gamma_feature = 2,
                 ):
        super().__init__()
        self.training_steps = training_steps
        self.eps = eps
        self.alpha_poisson = alpha_poisson
        self.lambda_feature = lambda_feature
        self.gamma_feature = gamma_feature


        self.encoder_layer0 = nn.Sequential(
            SineLayer(3, mid_channel),
            SineLayer(mid_channel, 100),
            SineLayer(100, mid_channel),
            SineLayer(mid_channel, 30),
            DenseLayer(30, hidden_dims[0])
        )

        # 避免 hidden_dims 越界
        if len(hidden_dims) >= 3:
            enc_out_dim = hidden_dims[2]
        else:
            enc_out_dim = hidden_dims[1]

        self.encoder_layer1 = DenseLayer(hidden_dims[0], enc_out_dim)

        self.decoder = nn.Sequential(
            SineLayer(enc_out_dim, mid_channel),
            DenseLayer(mid_channel, hidden_dims[0])
        )
        
        self.deconv_alpha_layer = DenseLayer(enc_out_dim + slice_emb_dim, 
                                             1, zero_init=True)
        self.deconv_beta_layer = nn.Sequential(
            DenseLayer(enc_out_dim, n_celltypes, zero_init=True)
        )

        self.gamma = nn.Parameter(torch.Tensor(n_slices, hidden_dims[0]).zero_())
        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

    def forward(self, 
                coord,
                node_feats,
                count_matrix,
                library_size,
                slice_label,
                basis,
                c,
                step
                ):
        # encoder
        self.node_feats = node_feats

        self.coord = coord / c

        #self.coord = coord
        Z, mid_fea = self.encoder(node_feats)

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)

        self.node_feats_recon = self.decoder(Z)

        # 期望 counts
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        expected_counts = library_size * lam

        # ========== 损失函数 ==========
        poisson_loss = -self.alpha_poisson * torch.mean(
            torch.sum(count_matrix * (torch.log(library_size + 1e-6) + log_lam) 
                      - expected_counts, dim=1)
        )

        self.decon_loss = poisson_loss
        # feature loss
        self.fea_loss = self.lambda_feature*torch.norm(node_feats-mid_fea, 2) + \
                        self.gamma_feature*torch.norm(node_feats-self.node_feats_recon, 2)
        
        loss = self.decon_loss +  self.fea_loss
        denoise = torch.matmul(beta, basis)
        return loss, mid_fea, denoise, Z, 0, 0


    def evaluate(self, coord, node_feats, slice_label):
        slice_label_emb = self.slice_emb(slice_label)
        Z, _ = self.encoder(node_feats)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        return Z, beta, alpha, self.gamma
            
    def encoder(self, H):
        self.mid_fea = self.encoder_layer0(self.coord)
        Z = self.encoder_layer1(self.mid_fea)
        return Z, self.mid_fea
    
    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(torch.sin(Z))
        beta = F.softmax(beta, dim=1)
        H = torch.sin(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha

    def inference_encoder(self, coord):
        """
        专门用于推理的 encoder 方法，直接接受坐标作为参数
        """
        coord_normalized = coord / 100  # 保持与训练时的归一化一致
        mid_fea = self.encoder_layer0(coord_normalized)
        Z = self.encoder_layer1(mid_fea)
        return Z, mid_fea

    def inference_with_coords(self, coord, slice_label):
        """
        使用坐标和切片标签进行推理
        """
        # 编码
        Z, mid_fea = self.inference_encoder(coord)
        
        # 解卷积
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        
        # 解码
        reconstructed = self.decoder(Z)
        

        return Z, beta, alpha, reconstructed
