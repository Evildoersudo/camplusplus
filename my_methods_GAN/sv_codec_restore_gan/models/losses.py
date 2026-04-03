from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def loss_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    noise = target - pred
    s_target = (target.pow(2).sum(dim=-1) + eps)
    s_noise = (noise.pow(2).sum(dim=-1) + eps)
    sdr = 10.0 * torch.log10(s_target / s_noise)
    return -sdr.mean()


def loss_lsd(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=pred.device)
    p = torch.stft(pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    t = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs()
    return torch.mean(torch.sqrt(torch.mean((_safe_log(p) - _safe_log(t)).pow(2), dim=-2)))


def loss_mag_real_imag(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 512, hop: int = 128):
    window = torch.hann_window(n_fft, device=pred.device)
    p = torch.stft(pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    t = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = F.l1_loss(p.abs(), t.abs())
    real = F.l1_loss(p.real, t.real)
    imag = F.l1_loss(p.imag, t.imag)
    return mag, real, imag


def loss_rec(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    l_sdr = loss_sdr(pred, target)
    l_lsd = loss_lsd(pred, target)
    l_mag, l_real, l_imag = loss_mag_real_imag(pred, target)
    total = 2.0 * l_sdr + 1.5 * l_lsd + 70.0 * l_mag + 30.0 * (l_real + l_imag)
    return {
        "total": total,
        "sdr": l_sdr,
        "lsd": l_lsd,
        "mag": l_mag,
        "real": l_real,
        "imag": l_imag,
    }


def _flatten_scores(disc_outs):
    return [score for score, _ in disc_outs]


def loss_adv_generator(fake_outs) -> torch.Tensor:
    losses = []
    for score in _flatten_scores(fake_outs):
        losses.append(F.mse_loss(score, torch.ones_like(score)))
    return torch.stack(losses).mean()


def loss_adv_discriminator(real_outs, fake_outs) -> torch.Tensor:
    losses = []
    for (real_score, _), (fake_score, _) in zip(real_outs, fake_outs):
        losses.append(F.mse_loss(real_score, torch.ones_like(real_score)))
        losses.append(F.mse_loss(fake_score, torch.zeros_like(fake_score)))
    return torch.stack(losses).mean()


def loss_feature_matching(real_outs, fake_outs) -> torch.Tensor:
    losses = []
    for (_, real_feats), (_, fake_feats) in zip(real_outs, fake_outs):
        for rf, ff in zip(real_feats, fake_feats):
            losses.append(F.l1_loss(ff, rf.detach()))
    if not losses:
        return torch.zeros((), device=fake_outs[0][0].device)
    return torch.stack(losses).mean()
