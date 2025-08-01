import torch
import torch.nn as nn


class PPA(nn.Module):
    """Panoramic Perceptual Attention for [B, N, D] input"""

    def __init__(self, dim, qk_dim=16, pdim=32):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        # LayerNorm 替换 GroupNorm，适配 [B, N, D]
        self.pre_norm = nn.LayerNorm(pdim)

        # 线性层替换 Conv2d_BN，生成 q, k, v
        self.qkv = nn.Linear(pdim, qk_dim * 2 + pdim)

        # 投影层，保持输出维度为 dim
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # x: [B, N, D], where N = H*W or H*W+1 (with CLS token), D = dim
        B, N, D = x.shape

        # 分割输入为 x1 (pdim 通道) 和 x2 (剩余通道)
        x1, x2 = torch.split(x, [self.pdim, self.dim - self.pdim], dim=-1)  # x1: [B, N, pdim], x2: [B, N, dim-pdim]

        # 归一化 x1
        x1 = self.pre_norm(x1)  # [B, N, pdim]

        # 计算 q, k, v
        qkv = self.qkv(x1)  # [B, N, qk_dim*2 + pdim]
        q, k, v = qkv.split([self.qk_dim, self.qk_dim, self.pdim], dim=-1)  # q, k: [B, N, qk_dim], v: [B, N, pdim]

        # 单头自注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, N, N]
        attn = attn.softmax(dim=-1)
        x1 = (attn @ v)  # [B, N, pdim]

        # 拼接 x1 和 x2
        x = torch.cat([x1, x2], dim=-1)  # [B, N, dim]

        # 投影到输出
        x = self.proj(x)  # [B, N, dim]

        return x