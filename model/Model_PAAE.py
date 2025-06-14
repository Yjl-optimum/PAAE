import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np
import os
from crossatt import WindowedSelfAttention, CrossAttentionBlock

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
torch.cuda.set_device(0)


class SpatialTransformerEncoder(nn.Module):
    def __init__(self, num_keypoints, feature_dim, n_head, dim_ff, num_layers, dropout=0.1):
        super(SpatialTransformerEncoder, self).__init__()
        d_model = feature_dim  # Use feature_dim directly for d_model
        self.encoder = TransformerEncoder(d_model, n_head, dim_ff, num_layers, dropout)

    def forward(self, x):
        batch_size, num_frames, num_keypoints, feature_dim = x.shape
        x_reshaped = x.view(batch_size * num_frames, num_keypoints, feature_dim)
        spatial_features = self.encoder(x_reshaped)
        spatial_features = spatial_features.view(batch_size, num_frames, num_keypoints, feature_dim)
        return spatial_features


class AutoEncoder(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(AutoEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(512, dropout_prob)
        self.ln = nn.LayerNorm(512)
        self.crossatt =nn.ModuleList([CrossAttentionBlock(
            dim=512,
            num_heads=4,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.2,
            attn_drop=0.1,
            drop_path=0.1
        ) for i in range(3)])
        self.windowatt = nn.ModuleList([WindowedSelfAttention(
            dim=512,
            num_heads=4,
            window_size=4,
        )for i in range(3)])
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout_prob),
        )
        self.decoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x):
        x_pos = self.pos_encoder(x)
        x1 = x_pos
        y = x1[:,32:96,:]
        for blk in self.crossatt:  ### cross-attention blocks
            x1 = blk(x1, y)
        for i, winatt in enumerate(self.windowatt):  ### self-attention blocks with windowed self-attention
            x1 = winatt(x1)
        x = self.ln(x+x1)
        x_encoder = self.encoder(x)
        x_decoder = self.decoder(x_encoder)
        return x_encoder, x_decoder


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        # Check if d_model is odd
        if d_model % 2 == 1:
            # Handle the last dimension separately for odd d_model
            pe[:, 1::2] = torch.cos(position * div_term)[:, :d_model // 2]
            pe[:, -1] = 0  # Set last dimension to zero
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiheadSelfAttention, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        q = self.q_linear(x).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.out_linear(attn_output)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadSelfAttention(d_model, n_head)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_head, dim_ff, dropout) for _ in range(num_layers)])
        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src):
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src)
        return src



class MultiheadSelfAttentionWithFourier(nn.Module):
    def __init__(self, d_model, n_head, use_fft=False, use_ifft=False):
        super(MultiheadSelfAttentionWithFourier, self).__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.use_fft = use_fft
        self.use_ifft = use_ifft

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        # print('x',x.shape)
        q = self.q_linear(x).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_length, self.n_head, self.d_k).transpose(1, 2)

        if self.use_fft:
            # Apply Fourier Transform to q, k, and v
            q = torch.fft.rfft(q, dim=-1, norm='ortho')
            k = torch.fft.rfft(k, dim=-1, norm='ortho')
            v = torch.fft.rfft(v, dim=-1, norm='ortho')


        # Compute attention scores in frequency domain
        scores_complex = q @ k.conj().transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores_magnitude = scores_complex.real

        # Attention weights in frequency domain
        attn_weights = torch.nn.functional.softmax(scores_magnitude, dim=-1)
        attn_weights = torch.complex(attn_weights, torch.zeros_like(attn_weights))


        # Apply attention weights to value in frequency domain
        attn_output = torch.matmul(attn_weights, v)

        # Optionally apply IFFT to bring back to time domain
        attn_output = torch.fft.irfft(attn_output, dim=-1, norm='ortho')

        # Reshape to original form
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        return self.out_linear(attn_output)




class TransformerEncoderLayerWithFourier(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, use_fft=False, use_ifft=False, dropout=0.1):
        super(TransformerEncoderLayerWithFourier, self).__init__()
        self.self_attn = MultiheadSelfAttentionWithFourier(d_model, n_head, use_fft, use_ifft)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.nn.functional.relu(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src#,qo


class TransformerEncoderWithFourier(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, num_layers, use_fft=False, use_ifft=False, dropout=0.1):
        super(TransformerEncoderWithFourier, self).__init__()

        # First layer with Fourier Transform
        self.layers = nn.ModuleList(
            [TransformerEncoderLayerWithFourier(d_model, n_head, dim_ff, use_fft=True, dropout=dropout)])

        # Middle layers without Fourier Transform
        self.layers.extend(
            [TransformerEncoderLayerWithFourier(d_model, n_head, dim_ff, use_fft=True, dropout=dropout) for _ in
             range(num_layers - 2)])

        # Last layer with Inverse Fourier Transform
        self.layers.append(TransformerEncoderLayerWithFourier(d_model, n_head, dim_ff, use_fft=True, dropout=dropout))

        self.pos_encoder = PositionalEncoding(d_model, dropout)

    def forward(self, src):
        for layer in self.layers:
            src= layer(src)
        return src


class OneDRAC_Pose(nn.Module):
    def __init__(self, num_frames):
        super(OneDRAC_Pose, self).__init__()
        self.num_frames = num_frames
        self.fc0 = nn.Sequential(
            nn.Linear(102, 512),
            nn.Dropout(0.2),
        )

        # Initialize the SpatialTransformerEncoder for spatial information extraction
        self.spatialTransformer = SpatialTransformerEncoder(17, 3, n_head=3, dim_ff=512,
                                                            num_layers=2, dropout=0.2)
        # Use FFT in the first layer and IFFT in the last layer of skeletonFeatureExtractor
        self.skeletonFeatureExtractor = TransformerEncoderWithFourier(
            d_model=512, n_head=1, dim_ff=2048, num_layers=6, dropout=0.4
        )

        self.AutoEncoder = AutoEncoder()

        # Use FFT in the first layer and IFFT in the last layer of transEncoder2
        self.transEncoder2 = TransformerEncoder(
            d_model=512, n_head=8, dim_ff=512, num_layers=1, dropout=0.2
        )

    def forward(self, x, ret_sims=True):
        batch_size, num_frames, num_keypoints, feature_dim = x.shape
        # Reshape x to [batch_size * num_frames, num_keypoints, feature_dim]
        x_reshaped = x.view(batch_size * num_frames, num_keypoints,
                            feature_dim)  # [batch_size * num_frames, num_keypoints, feature_dim]

        # Process all frames together
        spatial_features = self.spatialTransformer(x)  # [batch_size, num_frames, num_keypoints, feature_dim]

        # Flatten spatial features for further processing
        spatial_features = spatial_features.view(batch_size, num_frames, -1)

        # Concatenate with the original flattened features
        skeleton_features = torch.cat((x.view(batch_size, num_frames, -1), spatial_features), dim=-1)

        # print('3----------', skeleton_features.shape)
        skeleton_features = self.fc0(skeleton_features)
        reshaped_tensor = skeleton_features

        skeleton_features = self.skeletonFeatureExtractor(skeleton_features)
        x_encoder, x_decoder = self.AutoEncoder(skeleton_features)
        x_decoder = self.transEncoder2(x_decoder)

        if ret_sims:
            return reshaped_tensor, reshaped_tensor, x_encoder, x_decoder
        return x_encoder, x_decoder


