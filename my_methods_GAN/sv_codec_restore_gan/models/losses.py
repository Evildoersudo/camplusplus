from __future__ import annotations

import torch
import torch.nn.functional as F


def _safe_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x.clamp_min(eps))


def loss_si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    target_energy = target.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
    scale = (pred * target).sum(dim=-1, keepdim=True) / target_energy
    s_target = scale * target
    e_noise = pred - s_target
    ratio = s_target.pow(2).sum(dim=-1).clamp_min(eps) / e_noise.pow(2).sum(dim=-1).clamp_min(eps)
    return -10.0 * torch.log10(ratio).mean()


def loss_mrstft(
    pred: torch.Tensor,
    target: torch.Tensor,
    resolutions: tuple[tuple[int, int, int], ...] = ((256, 64, 256), (512, 128, 512), (1024, 256, 1024)),
    eps: float = 1e-7,
) -> torch.Tensor:
    losses = []
    for n_fft, hop, win in resolutions:
        window = torch.hann_window(win, device=pred.device)
        p = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        t = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)

        p_mag = p.abs().clamp_min(eps)
        t_mag = t.abs().clamp_min(eps)

        spectral_convergence = torch.linalg.norm(t_mag - p_mag) / torch.linalg.norm(t_mag)
        log_mag = F.l1_loss(_safe_log(p_mag, eps), _safe_log(t_mag, eps))
        losses.append(spectral_convergence + log_mag)
    return torch.stack(losses).mean()


def loss_complex_l1(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 512, hop: int = 128):
    window = torch.hann_window(n_fft, device=pred.device)
    p = torch.stft(pred, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    t = torch.stft(target, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    return F.l1_loss(p.real, t.real) + F.l1_loss(p.imag, t.imag)


def loss_rec(
    pred: torch.Tensor,
    target: torch.Tensor,
    si_sdr_weight: float = 1.0,
    mrstft_weight: float = 1.0,
    complex_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    l_si_sdr = loss_si_sdr(pred, target)
    l_mrstft = loss_mrstft(pred, target)
    l_complex = loss_complex_l1(pred, target)
    total = si_sdr_weight * l_si_sdr + mrstft_weight * l_mrstft + complex_weight * l_complex
    return {
        "total": total,
        "si_sdr": l_si_sdr,
        "mrstft": l_mrstft,
        "complex": l_complex,
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
