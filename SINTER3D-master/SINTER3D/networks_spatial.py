import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiScaleCoordEncoder(nn.Module):
    """Multi-scale coordinate encoder with improved coordinate handling"""
    def __init__(self, coord_dim=3, out_dim=64, num_freqs=10):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_freqs = num_freqs
        
        # Encode with multiple frequencies
        self.freqs = torch.nn.Parameter(
            torch.logspace(0, num_freqs-1, num_freqs, base=2.0), 
            requires_grad=False
        )
        
        # Encode XY and Z separately using different networks
        xy_encoding_dim = 2 * num_freqs * 2  # 2 coordinates * num_freqs * (sin+cos)
        z_encoding_dim = 1 * num_freqs * 2   # 1 coordinate * num_freqs * (sin+cos)
        
        self.xy_encoder = nn.Sequential(
            nn.Linear(xy_encoding_dim, out_dim//2),
            nn.ReLU(),
            nn.Linear(out_dim//2, out_dim//2)
        )
        
        self.z_encoder = nn.Sequential(
            nn.Linear(z_encoding_dim, out_dim//4), 
            nn.ReLU(),
            nn.Linear(out_dim//4, out_dim//4)
        )
        
        # Fusion layer
        self.fusion = nn.Linear(out_dim//2 + out_dim//4, out_dim)
    
    def forward(self, coords):
        """
        coords: [N, 3] - Normalized coordinates (x, y, z)
        """
        xy_coords = coords[:, :2]  # [N, 2]
        z_coords = coords[:, 2:3]  # [N, 1]
        
        # Multi-frequency positional encoding
        xy_encoded = self.positional_encoding(xy_coords, self.freqs)
        z_encoded = self.positional_encoding(z_coords, self.freqs) 
        
        # Process XY and Z separately
        xy_features = self.xy_encoder(xy_encoded)
        z_features = self.z_encoder(z_encoded)
        
        # Fuse features
        combined = torch.cat([xy_features, z_features], dim=1)
        output = self.fusion(combined)
        
        return output
    
    def positional_encoding(self, coords, freqs):
        """Positional encoding: sin(freq * coord), cos(freq * coord)"""
        # coords: [N, coord_dim], freqs: [num_freqs]
        coords_expanded = coords.unsqueeze(-1)  # [N, coord_dim, 1]
        freqs_expanded = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, num_freqs]
        
        angles = coords_expanded * freqs_expanded  # [N, coord_dim, num_freqs]
        
        # Compute sin and cos features
        sin_encoding = torch.sin(angles)  # [N, coord_dim, num_freqs]
        cos_encoding = torch.cos(angles)  # [N, coord_dim, num_freqs]
        
        # Concatenate and flatten
        encoding = torch.stack([sin_encoding, cos_encoding], dim=-1)  # [N, coord_dim, num_freqs, 2]
        encoding = encoding.view(coords.shape[0], -1)  # [N, coord_dim*num_freqs*2]
        
        return encoding


class DenseLayer(nn.Module):
    def __init__(self, c_in, c_out, zero_init=False):
        super().__init__()
        self.linear = nn.Linear(c_in, c_out)
        
        if zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            nn.init.uniform_(self.linear.weight.data, -np.sqrt(
                6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, node_feats):
        node_feats = self.linear(node_feats)
        return node_feats


class ImprovedSineLayer(nn.Module):
    """Improved SineLayer with a better initialization strategy."""
    def __init__(self, c_in, c_out, bias=True, zero_init=False, omega_0=1, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.zero_init = zero_init
        self.is_first = is_first
        self.in_features = c_in
        self.linear = nn.Linear(c_in, c_out, bias=bias)
        
        if self.zero_init:
            nn.init.zeros_(self.linear.weight.data)
        else:
            if self.is_first:
                # Use uniform initialization for the first layer
                bound = 1 / c_in
                self.linear.weight.data.uniform_(-bound, bound)
            else:
                # Use SIREN-compatible initialization for subsequent layers
                bound = np.sqrt(6 / c_in) / self.omega_0
                self.linear.weight.data.uniform_(-bound, bound)
            
            if bias:
                nn.init.zeros_(self.linear.bias.data)
            
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class ResidualSineBlock(nn.Module):
    """Sine block with residual connections."""
    def __init__(self, dim, omega_0=1, dropout=0.1):
        super().__init__()
        self.layer1 = ImprovedSineLayer(dim, dim, omega_0=omega_0)
        self.layer2 = ImprovedSineLayer(dim, dim, omega_0=omega_0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        return out + residual  # Residual connection


class SineLayer(ImprovedSineLayer):
    def __init__(self, c_in, c_out, bias=True, zero_init=False, omega_0=1):
        super().__init__(c_in, c_out, bias, zero_init, omega_0, is_first=False)


class DeconvNet(nn.Module):
    """Improved coordinate encoding"""
    def __init__(self, 
                 hidden_dims,
                 n_celltypes, 
                 n_slices,
                 slice_emb_dim, 
                 training_steps,
                 eps=1e-6,
                 mid_channel=200,
                 coord_emb_dim=64,  
                 num_freqs=12,      
                 c = 100,
                 alpha_poisson = 5,
                 lambda_feature = 1,
                 gamma_feature = 2
                 ):
        super().__init__()

        self.training_steps = training_steps
        self.eps = eps
        self.coord_emb_dim = coord_emb_dim
        self.c = c,
        self.alpha_poisson = alpha_poisson
        self.lambda_feature = lambda_feature
        self.gamma_feature = gamma_feature

        # Added: multi-scale coordinate encoder
        self.coord_encoder = MultiScaleCoordEncoder(
            coord_dim=3, 
            out_dim=coord_emb_dim,
            num_freqs=num_freqs
        )

        if len(hidden_dims) >= 3:
            enc_out_dim = hidden_dims[2]
        else:
            enc_out_dim = hidden_dims[1]

        progressive_dims = self._design_progressive_dims(coord_emb_dim, enc_out_dim, num_layers=4)
        
        # Build the progressive encoder
        encoder_layers = []
        for i in range(len(progressive_dims) - 1):
            in_dim = progressive_dims[i]
            out_dim = progressive_dims[i + 1]
            
            if i == 0:
                # First layer
                encoder_layers.append(
                    ImprovedSineLayer(in_dim, out_dim, is_first=True, omega_0=1)
                )
            else:
                # Middle layers: Sine + residual block
                encoder_layers.append(
                    ImprovedSineLayer(in_dim, out_dim, omega_0=1)
                )

                if out_dim >= 64:
                    encoder_layers.append(
                        ResidualSineBlock(out_dim, omega_0=1, dropout=0.1)
                    )
        
        self.progressive_encoder = nn.ModuleList(encoder_layers)
        
        # Save dimension information for forward()
        self.progressive_dims = progressive_dims
        
        final_enc_dim = progressive_dims[-1]
        if final_enc_dim != hidden_dims[0]:
            self.mid_fea_adapter = nn.Linear(final_enc_dim, hidden_dims[0])
        else:
            self.mid_fea_adapter = None

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

    def _design_progressive_dims(self, start_dim, end_dim, num_layers=4):
        """Design the progressive dimension schedule."""
        if num_layers < 2:
            return [start_dim, end_dim]

        peak_dim = max(start_dim, end_dim, 256)  
        
        if num_layers == 2:
            return [start_dim, end_dim]
        elif num_layers == 3:
            return [start_dim, peak_dim, end_dim]
        else:

            dims = []
            dims.append(start_dim)
        
            mid_layers = num_layers - 2
            first_half = mid_layers // 2
            second_half = mid_layers - first_half
            
            # Ascending phase
            if first_half > 0:
                step = (peak_dim - start_dim) / (first_half + 1)
                for i in range(first_half):
                    dims.append(int(start_dim + step * (i + 1)))
            
            # Add the peak value
            if peak_dim not in dims:
                dims.append(peak_dim)
            
            # Descending phase
            if second_half > 0:
                step = (peak_dim - end_dim) / (second_half + 1)
                for i in range(second_half):
                    dims.append(int(peak_dim - step * (i + 1)))
            
            dims.append(end_dim)
            
            # Ensure valid dimensions
            dims = [max(d, 32) for d in dims] 
            
            return dims

    def forward(self, 
                coord,
                node_feats,
                count_matrix,
                library_size,
                slice_label,
                basis,
                c,
                step):
        # Store the original input
        self.node_feats = node_feats
        self.c = c
        self.coord = coord / self.c

        coord_features = self.coord_encoder(self.coord)  # [N, coord_emb_dim]

        # Encoder now uses coordinate features instead of raw coordinates
        Z, mid_fea = self.encoder(coord_features)  

        # deconvolutioner
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)

        self.node_feats_recon = self.decoder(Z)

        # Expected counts
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma[slice_label]
        lam = torch.exp(log_lam)
        expected_counts = library_size * lam

        # ========== Loss function ==========
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
        
        # Use coordinate encoding
        coord_normalized = coord / self.c
        coord_features = self.coord_encoder(coord_normalized)
        Z, _ = self.encoder(coord_features)  # Modified: use coordinate features
        
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        return Z, beta, alpha, self.gamma
            
    def encoder(self, coord_features):  
        """Improved progressive encoder"""
        x = coord_features
        
        for layer in self.progressive_encoder:
            x = layer(x)
        
        # Handle mid_fea compatibility for loss computation
        if self.mid_fea_adapter is not None:
            self.mid_fea = self.mid_fea_adapter(x)
        else:
            self.mid_fea = x
            
        return x, self.mid_fea
    
    def deconvolutioner(self, Z, slice_label_emb):
        beta = self.deconv_beta_layer(torch.sin(Z))
        beta = F.softmax(beta, dim=1)
        H = torch.sin(torch.cat((Z, slice_label_emb), axis=1))
        alpha = self.deconv_alpha_layer(H)
        return beta, alpha

    def inference_encoder(self, coord):
        """Encoder used during inference"""
        coord_normalized = coord / self.c
        coord_features = self.coord_encoder(coord_normalized)
        Z, mid_fea = self.encoder(coord_features)
        return Z, mid_fea

    def inference_with_coords(self, coord, slice_label):
        """Run inference using coordinates and slice labels."""
        # Encode
        Z, mid_fea = self.inference_encoder(coord)
        
        # Deconvolution
        slice_label_emb = self.slice_emb(slice_label)
        beta, alpha = self.deconvolutioner(Z, slice_label_emb)
        
        # Decode
        reconstructed = self.decoder(Z)
        
        return Z, beta, alpha, reconstructed


