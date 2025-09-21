import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class WFF(nn.Module):
    def __init__(self, channel=256, num_heads=8):
        super(WFF, self).__init__()
        self.channel = channel
        self.num_heads = num_heads

        # 加权系数生成模块
        self.fc = nn.Sequential(
            nn.Linear(2 * channel, channel),
            nn.ReLU(),
            nn.Linear(channel, 1),
            nn.Sigmoid()
        )

        # 自注意力机制
        self.self_attn_pseudo = nn.MultiheadAttention(channel, num_heads, batch_first=True)
        self.self_attn_reg = nn.MultiheadAttention(channel, num_heads, batch_first=True)

        # 层归一化
        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)

        # 噪声抑制相关参数
        self.conv_noise = nn.Conv2d(channel, 1, 3, 1, 1)

    def generate_weight_coefficient(self, F_pseudo, F_reg):
        """生成加权系数α"""
        # 全局平均池化获取全局特征
        gap_pseudo = F.adaptive_avg_pool2d(F_pseudo, (1, 1)).squeeze(-1).squeeze(-1)
        gap_reg = F.adaptive_avg_pool2d(F_reg, (1, 1)).squeeze(-1).squeeze(-1)

        # 拼接特征
        combined = torch.cat([gap_pseudo, gap_reg], dim=1)

        # 通过全连接层和sigmoid生成加权系数
        alpha = self.fc(combined)
        return alpha.unsqueeze(-1).unsqueeze(-1)  # 恢复空间维度

    def self_attention_processing(self, F_pseudo, F_reg):
        """自注意力处理"""
        batch_size, channels, h, w = F_pseudo.shape

        # 重塑为序列形式 (B, H*W, C)
        F_pseudo_flat = F_pseudo.view(batch_size, channels, -1).permute(0, 2, 1)
        F_reg_flat = F_reg.view(batch_size, channels, -1).permute(0, 2, 1)

        # 伪标签特征自注意力
        Attn_pseudo, _ = self.self_attn_pseudo(F_pseudo_flat, F_pseudo_flat, F_pseudo_flat)
        Attn_pseudo = self.norm1(Attn_pseudo + F_pseudo_flat)
        Attn_pseudo = Attn_pseudo.permute(0, 2, 1).view(batch_size, channels, h, w)

        # 图像特征自注意力
        Attn_reg, _ = self.self_attn_reg(F_reg_flat, F_reg_flat, F_reg_flat)
        Attn_reg = self.norm2(Attn_reg + F_reg_flat)
        Attn_reg = Attn_reg.permute(0, 2, 1).view(batch_size, channels, h, w)

        return Attn_pseudo, Attn_reg

    def noise_suppression(self, F_pseudo):
        """噪声抑制与边界增强"""
        batch_size, channels, h, w = F_pseudo.shape

        # 计算空间一致性权重
        # 使用3x3卷积近似邻域操作
        unfolded = F.unfold(F_pseudo, kernel_size=3, padding=1, stride=1)
        unfolded = unfolded.view(batch_size, channels, 9, h, w)

        # 计算中心像素与邻域像素的差异
        center = F_pseudo.unsqueeze(2)
        diff = unfolded - center
        variance = torch.mean(diff.pow(2), dim=2)  # 计算方差

        # 计算权重：方差越小（一致性越高），权重越大
        W_alpha = torch.exp(-variance.mean(dim=1, keepdim=True))  # 跨通道平均

        # 应用噪声抑制
        F_noise_suppressed = F_pseudo * W_alpha

        return F_noise_suppressed

    def forward(self, F_pseudo, F_reg):
        """
        Args:
            F_pseudo: 伪标签特征, shape [B, C, H, W]
            F_reg: 图像编码特征, shape [B, C, H, W]
        Returns:
            F_fused: 融合后的特征, shape [B, C, H, W]
        """

        # 步骤1: 生成加权系数
        alpha = self.generate_weight_coefficient(F_pseudo, F_reg)

        # 步骤2: 自注意力处理
        Attn_pseudo, Attn_reg = self.self_attention_processing(F_pseudo, F_reg)

        # 步骤3: 加权融合
        F_fused = alpha * Attn_pseudo + (1 - alpha) * Attn_reg

        # 步骤4: 噪声抑制与边界增强
        F_fused = self.noise_suppression(F_fused)

        return F_fused