"""CA-AFC 前端网络定义。

本模块实现 `my_methods/note/method_way.md` 中描述的前端补偿网络。
整体思路不是直接替代 CAM++，而是放在 CAM++ 前面，先把经过 codec
压缩失真的 FBank 特征做补偿，再把补偿后的特征送入冻结的 CAM++ 后端。

网络主流程如下：

1. 主分支对 codec 退化后的 FBank 做时频卷积编码。
2. 辅助分支对 pitch / delta-pitch / voiced-flag 做时间建模。
3. 用门控融合模块自适应融合两路隐藏表示。
4. 预测逐帧逐频带的 band attention，重标定原始 FBank。
5. 再预测一个 residual，对失真部分做残差补偿。
6. 输出增强后的 80 维 FBank 序列，供后续 CAM++ 提取说话人嵌入。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class FrontendOutput:
    """前端输出结构体。

    字段说明：
    - enhanced: 最终增强后的 FBank，形状为 [B, T, 80]
    - band_weights: 逐帧逐频带权重，形状为 [B, T, 80]
    - residual: 残差补偿项，形状为 [B, T, 80]
    - fused_hidden: 主分支和辅分支融合后的隐藏表示，供分析和可视化使用
    """

    enhanced: torch.Tensor
    band_weights: torch.Tensor
    residual: torch.Tensor
    fused_hidden: torch.Tensor


class SpectralEncoder(nn.Module):
    """主分支：对 codec 退化后的 FBank 做浅层时频卷积编码。

    输入形状：
    - x: [B, T, 80]

    处理过程：
    - 先扩成 [B, 1, T, 80]，把 FBank 视为单通道时频图
    - 通过多层 2D 卷积提取局部时频纹理
    - 最后在频率维做平均池化，只保留按时间展开的隐藏表示

    输出形状：
    - [B, T, hidden_dim]

    设计目的：
    - 主分支重点学习 codec 压缩对频谱结构造成的失真模式
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, hidden_dim, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 80] -> [B, 1, T, 80]
        feat = self.net(x.unsqueeze(1))
        # 在频率维求均值，把卷积特征压成时间序列隐藏表示
        return feat.mean(dim=-1).transpose(1, 2)


class AuxEncoder(nn.Module):
    """辅助分支：对 pitch/voicing 一类辅助时序特征做编码。

    当前辅助特征维度通常为 3：
    - pitch
    - delta-pitch
    - voiced flag

    输入形状：
    - aux: [B, T, 3]

    输出形状：
    - [B, T, hidden_dim]

    设计目的：
    - 为主分支提供额外的时间上下文线索
    - 帮助网络判断哪些时刻的语音更稳定、哪些时刻更可信
    """

    def __init__(self, aux_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(aux_dim, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, aux: torch.Tensor) -> torch.Tensor:
        hidden = self.net(aux.transpose(1, 2)).transpose(1, 2)
        return self.norm(hidden)


class AttentiveFusion(nn.Module):
    """门控融合模块：自适应融合主分支与辅助分支。

    融合方式不是简单相加，而是学习一个门控系数 alpha：

    - alpha 越大，越依赖主分支的谱表示
    - alpha 越小，越依赖辅助分支的时间线索

    这样可以让网络针对不同时间帧动态调整信息来源。
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.aux_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, spectral_hidden: torch.Tensor, aux_hidden: torch.Tensor) -> torch.Tensor:
        # 先把辅助分支映射到与主分支同维度的空间
        aux_proj = self.aux_proj(aux_hidden)
        # 根据两路特征共同预测门控系数 alpha
        alpha = torch.sigmoid(self.gate(torch.cat([spectral_hidden, aux_hidden], dim=-1)))
        # 动态融合：按帧决定更相信谱信息还是辅助信息
        return alpha * spectral_hidden + (1.0 - alpha) * aux_proj


class CAAFCFrontend(nn.Module):
    """CA-AFC 前端主体。

    输入：
    - codec_fbank: [B, T, 80]
      经过 codec 压缩退化后的 FBank 特征
    - aux_feats: [B, T, d_aux]
      与 codec 语音对应的辅助时序特征

    输出：
    - enhanced: [B, T, 80]
      增强后的 FBank，时间长度和特征维度与输入保持一致

    整体结构：
    1. SpectralEncoder 提取主分支谱表示
    2. AuxEncoder 提取辅助分支时间表示
    3. AttentiveFusion 做门控融合
    4. band_attention 预测逐频带权重
    5. residual_head 预测残差补偿
    6. enhanced = band_weights * codec_fbank + residual

    这样设计的好处是：
    - 保留原始 codec FBank 的主体结构
    - 只对受损频带做重标定和补偿
    - 输出仍然是标准 80 维 FBank，方便无缝接到 CAM++ 后端
    """

    def __init__(
        self,
        feat_dim: int = 80,
        aux_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        band_scale: float = 1.0,
        residual_scale: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.band_scale = float(band_scale)
        self.residual_scale = float(residual_scale)

        self.spectral_encoder = SpectralEncoder(hidden_dim=hidden_dim)
        self.aux_encoder = AuxEncoder(aux_dim=aux_dim, hidden_dim=hidden_dim)
        self.fusion = AttentiveFusion(hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 频带注意力头：先输出 logits，再映射为以 1.0 为中心的缩放系数。
        self.band_attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )
        # 残差补偿头：学习在加权后的 FBank 上额外补偿多少
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(self, codec_fbank: torch.Tensor, aux_feats: torch.Tensor) -> FrontendOutput:
        # 主分支：建模 codec 对频谱结构的影响
        spectral_hidden = self.spectral_encoder(codec_fbank)
        # 辅助分支：建模 pitch / voicing 等时间线索
        aux_hidden = self.aux_encoder(aux_feats)
        # 门控融合：按帧动态整合两路信息
        fused_hidden = self.dropout(self.fusion(spectral_hidden, aux_hidden))

        # 逐帧逐带权重，用于重标定原始退化 FBank
        band_logits = self.band_attention(fused_hidden)
        band_weights = 1.0 + self.band_scale * torch.tanh(band_logits)
        # 残差项，用于补偿仅靠缩放无法恢复的失真
        residual = self.residual_scale * self.residual_head(fused_hidden)
        # 先做 band-wise reweight，再加 residual 形成最终增强特征
        weighted = band_weights * codec_fbank
        enhanced = weighted + residual

        return FrontendOutput(
            enhanced=enhanced,
            band_weights=band_weights,
            residual=residual,
            fused_hidden=fused_hidden,
        )
