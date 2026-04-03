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
    time_weights: torch.Tensor
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
    - tf_hidden: [B, hidden_dim, T, F]
    - pooled_hidden: [B, T, hidden_dim]

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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, 80] -> [B, 1, T, 80]
        feat = self.net(x.unsqueeze(1))
        # pooled_hidden 在频率维求均值，供融合与残差分支使用。
        pooled_hidden = feat.mean(dim=-1).transpose(1, 2)
        return feat, pooled_hidden


class CodecConditionModulation(nn.Module):
    """Codec 条件调制模块（FiLM 风格）。"""

    def __init__(self, hidden_dim: int = 64, num_codecs: int = 8, codec_emb_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(num_codecs, codec_emb_dim)
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(codec_emb_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, pooled_hidden: torch.Tensor, tf_hidden: torch.Tensor, codec_ids: torch.Tensor | None):
        if codec_ids is None:
            return pooled_hidden, tf_hidden
        codec_embed = self.embedding(codec_ids)
        gamma_beta = self.to_gamma_beta(codec_embed)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        # Use tanh-bounded affine factors for stable conditioning.
        gamma = 1.0 + 0.5 * torch.tanh(gamma)
        beta = 0.5 * torch.tanh(beta)

        pooled_mod = gamma.unsqueeze(1) * pooled_hidden + beta.unsqueeze(1)
        tf_mod = gamma.unsqueeze(-1).unsqueeze(-1) * tf_hidden + beta.unsqueeze(-1).unsqueeze(-1)
        return pooled_mod, tf_mod


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


class TimeFrequencyAttention(nn.Module):
    """Generate frame-wise frequency and temporal reliability weights."""

    def __init__(self, hidden_dim: int = 64, feat_dim: int = 80, time_scale: float = 0.0):
        super().__init__()
        self.time_scale = float(time_scale)
        self.aux_to_chan = nn.Linear(hidden_dim, hidden_dim)
        self.aux_to_band = nn.Linear(hidden_dim, feat_dim)
        self.band_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        # Learnable frequency positional bias to stabilize per-band weighting.
        self.freq_pos_bias = nn.Parameter(torch.zeros(1, 1, feat_dim))
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tf_hidden: torch.Tensor,
        fused_hidden: torch.Tensor,
        aux_hidden: torch.Tensor,
        band_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aux_bias = self.aux_to_chan(aux_hidden).transpose(1, 2).unsqueeze(-1)
        band_logits = self.band_head(tf_hidden + aux_bias).squeeze(1)
        band_logits = band_logits + self.aux_to_band(aux_hidden) + self.freq_pos_bias
        band_weights = 1.0 + band_scale * torch.tanh(band_logits)

        time_logits = self.time_head(torch.cat([fused_hidden, aux_hidden], dim=-1))
        # Residual-style temporal reliability keeps weights close to 1 by default.
        time_weights = 1.0 + self.time_scale * torch.tanh(time_logits)
        return band_weights, time_weights


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
    6. enhanced = codec_fbank + (band_delta + residual) * time_weights

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
        band_scale: float = 0.3,
        residual_scale: float = 0.05,
        band_residual_scale: float = 1.0,
        time_scale: float = 0.0,
        num_codecs: int = 8,
        codec_emb_dim: int = 16,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.band_scale = float(band_scale)
        self.residual_scale = float(residual_scale)
        self.band_residual_scale = float(band_residual_scale)

        self.spectral_encoder = SpectralEncoder(hidden_dim=hidden_dim)
        self.codec_modulation = CodecConditionModulation(
            hidden_dim=hidden_dim,
            num_codecs=num_codecs,
            codec_emb_dim=codec_emb_dim,
        )
        self.aux_encoder = AuxEncoder(aux_dim=aux_dim, hidden_dim=hidden_dim)
        self.fusion = AttentiveFusion(hidden_dim=hidden_dim)
        self.tf_attention = TimeFrequencyAttention(
            hidden_dim=hidden_dim,
            feat_dim=feat_dim,
            time_scale=time_scale,
        )
        self.dropout = nn.Dropout(dropout)

        # 残差补偿头：学习在加权后的 FBank 上额外补偿多少
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feat_dim),
        )

    def forward(
        self,
        codec_fbank: torch.Tensor,
        aux_feats: torch.Tensor,
        codec_ids: torch.Tensor | None = None,
    ) -> FrontendOutput:
        # 主分支：建模 codec 对频谱结构的影响
        tf_hidden, spectral_hidden = self.spectral_encoder(codec_fbank)
        # 辅助分支：建模 pitch / voicing 等时间线索
        aux_hidden = self.aux_encoder(aux_feats)
        # 先做 codec 条件调制，再进行跨分支融合。
        spectral_hidden, tf_hidden = self.codec_modulation(spectral_hidden, tf_hidden, codec_ids)
        # 门控融合：按帧动态整合两路信息
        fused_hidden = self.dropout(self.fusion(spectral_hidden, aux_hidden))

        # 分离建模频带权重与时间可靠性权重。
        band_weights, time_weights = self.tf_attention(
            tf_hidden,
            fused_hidden,
            aux_hidden,
            band_scale=self.band_scale,
        )
        # 残差项，用于补偿仅靠缩放无法恢复的失真
        residual = self.residual_scale * self.residual_head(fused_hidden)
        # Use conservative delta-style enhancement to avoid damaging clean speech.
        band_delta = self.band_residual_scale * (band_weights - 1.0) * codec_fbank
        delta = (band_delta + residual) * time_weights
        enhanced = codec_fbank + delta

        return FrontendOutput(
            enhanced=enhanced,
            band_weights=band_weights,
            time_weights=time_weights,
            residual=residual,
            fused_hidden=fused_hidden,
        )
