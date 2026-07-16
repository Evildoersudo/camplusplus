from __future__ import annotations

from torch import nn
import torch.nn.init as init


def init_weights(module: nn.Module, nonlinearity: str = "relu", negative_slope: float = 0.0) -> None:
    """Initialize model parameters with commonly used, activation-aware schemes.

    Conv/Linear: Kaiming (for ReLU-like activations)
    Norm: weight=1, bias=0
    RNNs: weight_ih Xavier, weight_hh Orthogonal, bias=0
    MultiheadAttention: Xavier for projection weights, zero bias
    """
    # 1) Convolution / Linear blocks
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity=nonlinearity,
            a=negative_slope,
        )
        if module.bias is not None:
            init.constant_(module.bias, 0.0)
        return

    # 2) Normalization layers
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
        if module.weight is not None:
            init.constant_(module.weight, 1.0)
        if module.bias is not None:
            init.constant_(module.bias, 0.0)
        return

    # 3) MultiheadAttention
    if isinstance(module, nn.MultiheadAttention):
        for name, param in module.named_parameters(recurse=False):
            if "weight" in name:
                init.xavier_uniform_(param)
            elif "bias" in name:
                init.constant_(param, 0.0)
        return

    # 4) Recurrent layers
    if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
        for name, param in module.named_parameters(recurse=False):
            if "weight_ih" in name:
                init.xavier_uniform_(param)
            elif "weight_hh" in name:
                init.orthogonal_(param)
            elif "bias" in name:
                init.constant_(param, 0.0)


def apply_model_init(model: nn.Module, nonlinearity: str = "relu", negative_slope: float = 0.0) -> None:
    model.apply(lambda m: init_weights(m, nonlinearity=nonlinearity, negative_slope=negative_slope))
