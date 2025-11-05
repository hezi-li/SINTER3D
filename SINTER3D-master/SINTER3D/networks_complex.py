import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        return self.linear(x)


class SineLayer(nn.Module):
    def __init__(self, c_in, c_out, bias=True, zero_init=False, omega_0=1):
        super().__init__()
        self.omega_0 = omega_0
        self.zero_init = zero_init
        self.linear = nn.Linear(c_in, c_out, bias=bias)

        if self.zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data,
                             -np.sqrt(6 / (c_in + c_out)),
                             np.sqrt(6 / (c_in + c_out)))
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
                 gamma_feature = 2
                 ):
        super().__init__()
        self.training_steps = training_steps
        self.eps = eps
        self.alpha_poisson = alpha_poisson
        self.lambda_feature = lambda_feature
        self.gamma_feature = gamma_feature

        if len(hidden_dims) >= 3:
            enc_out_dim = hidden_dims[2]
        else:
            enc_out_dim = hidden_dims[1]

        # Teacher encoder (coord + gene)
        self.coord_encoder = nn.Sequential(
            SineLayer(3, mid_channel),
            SineLayer(mid_channel, 100),
            SineLayer(100, mid_channel),
            SineLayer(mid_channel, 30)
        )
        self.gene_encoder = DenseLayer(hidden_dims[0], hidden_dims[0])
        self.fusion_layer = DenseLayer(hidden_dims[0] + 30, enc_out_dim)

        # Student encoder (coord only)
        self.coord_latent_predictor = nn.Sequential(
            SineLayer(3, mid_channel),
            SineLayer(mid_channel, enc_out_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            SineLayer(enc_out_dim, mid_channel),
            DenseLayer(mid_channel, hidden_dims[0])
        )

        # Deconvolution
        self.deconv_alpha_layer = DenseLayer(enc_out_dim + slice_emb_dim, 1, zero_init=True)
        self.deconv_beta_layer = nn.Sequential(DenseLayer(enc_out_dim, n_celltypes, zero_init=True))

        self.gamma = nn.Parameter(torch.Tensor(n_slices, hidden_dims[0]).zero_())
        self.slice_emb = nn.Embedding(n_slices, slice_emb_dim)

    def forward(self, coord, node_feats, count_matrix, library_size, slice_label, basis, c, step):
        self.c = c
        self.coord = coord / c

        # Teacher branch: coord + gene
        coord_fea = self.coord_encoder(self.coord)  # (N, 30)
        gene_fea = self.gene_encoder(node_feats)    # (N, hidden_dims[0])
        fusion_input = torch.cat([gene_fea, coord_fea], dim=1)  # concat
        Z_teacher = self.fusion_layer(fusion_input)

        # Student branch: coord only
        Z_student = self.coord_latent_predictor(self.coord)

        # Deconvolution 使用 teacher 的 latent
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z_teacher, slice_label_emb)
        self.node_feats_recon = self.decoder(Z_teacher)

        # Loss 部分
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        expected_counts = library_size * lam

        poisson_loss = -self.alpha_poisson * torch.mean(
            torch.sum(count_matrix * (torch.log(library_size + 1e-6) + log_lam) - expected_counts, dim=1)
        )
        self.decon_loss = poisson_loss
        self.fea_loss = self.lambda_feature*torch.norm(node_feats - gene_fea, 2) + \
                        self.gamma_feature*torch.norm(node_feats - self.node_feats_recon, 2)

        # CoordLoss
        coord_loss = F.mse_loss(Z_student, Z_teacher)

        loss = self.decon_loss + self.fea_loss + coord_loss
        denoise = torch.matmul(beta, basis)

        return loss, Z_teacher, denoise,  coord_loss

    def evaluate(self, coord, node_feats, slice_label):
        self.coord = coord / self.c
        coord_fea = self.coord_encoder(self.coord)
        gene_fea = self.gene_encoder(node_feats)
        fusion_input = torch.cat([gene_fea, coord_fea], dim=1)
        Z_teacher = self.fusion_layer(fusion_input)

        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z_teacher, slice_label_emb)
        return Z_teacher, beta, alpha, self.gamma

    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(torch.sin(Z))
        beta = F.softmax(beta, dim=1)
        H = torch.sin(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha

    def inference_encoder(self, coord, coord_only=False):
        coord_norm = coord / self.c
        if coord_only:
            Z = self.coord_latent_predictor(coord_norm)
            return Z, None
        else:
            coord_fea = self.coord_encoder(coord_norm)
            # 由于推理时没有 gene 数据，所以 teacher 不能用
            return self.coord_latent_predictor(coord_norm), None
