"""v1 — PyTorch port of lambertae/drifting drift_loss.

Translated from drifting_ref/drift_loss.py (JAX). Semantics preserved:
 - per-temperature softmax normalization in both directions
 - geometric-mean affinity = sqrt(softmax_row * softmax_col)
 - diag mask blocks self-attraction
 - force = aff_pos * sum(aff_neg) - aff_neg * sum(aff_pos)
 - force across R is normalized by sqrt(f_norm) and summed
 - final loss = mean((gen_scaled - sg(gen_scaled + force))^2)
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
import torch.nn.functional as F


def _cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: [B, N, D], y: [B, M, D] -> [B, N, M]
    xy = torch.einsum("bnd,bmd->bnm", x, y)
    xn = torch.einsum("bnd,bnd->bn", x, x)
    yn = torch.einsum("bmd,bmd->bm", y, y)
    sq = xn.unsqueeze(-1) + yn.unsqueeze(1) - 2 * xy
    return torch.sqrt(torch.clamp(sq, min=eps))


def drift_loss(
    gen: torch.Tensor,              # [B, Cg, S]
    fixed_pos: torch.Tensor,        # [B, Cp, S]
    fixed_neg: Optional[torch.Tensor] = None,  # [B, Cn, S]
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
    topk_pos: int = 0,
    topk_neg: int = 0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Returns (loss[B], info_dict). loss is mean over batch-elements shape [B]."""
    B, Cg, S = gen.shape
    Cp = fixed_pos.shape[1]
    device = gen.device
    dtype = torch.float32

    if fixed_neg is None:
        fixed_neg = gen.new_zeros((B, 0, S))
    Cn = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones((B, Cg))
    if weight_pos is None:
        weight_pos = gen.new_ones((B, Cp))
    if weight_neg is None:
        weight_neg = gen.new_ones((B, Cn))

    gen = gen.to(dtype)
    fixed_pos = fixed_pos.to(dtype)
    fixed_neg = fixed_neg.to(dtype)
    weight_gen = weight_gen.to(dtype)
    weight_pos = weight_pos.to(dtype)
    weight_neg = weight_neg.to(dtype)

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)  # [B, Cg+Cn+Cp, S]
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)  # [B, Cg+Cn+Cp]

    with torch.no_grad():
        dist = _cdist(old_gen, targets)  # [B, Cg, T]
        weighted_dist = dist * targets_w.unsqueeze(1)
        scale = weighted_dist.mean() / targets_w.mean().clamp(min=1e-6)
        info: Dict[str, torch.Tensor] = {"scale": scale.detach()}

        scale_inputs = torch.clamp(scale / (S ** 0.5), min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs
        dist_normed = dist / torch.clamp(scale, min=1e-3)

        # diag mask: block self
        mask_val = 100.0
        diag = torch.eye(Cg, dtype=dtype, device=device)  # [Cg, Cg]
        block_mask = F.pad(diag, (0, Cn + Cp))  # [Cg, Cg+Cn+Cp]
        dist_normed = dist_normed + block_mask.unsqueeze(0) * mask_val

        split_idx = Cg + Cn
        force_across_R = torch.zeros_like(old_gen_scaled)
        for R in R_list:
            logits = -dist_normed / R
            # top-K mask: keep only top-K nearest pos and top-K nearest neg
            # per gen row, suppress the rest with a large negative bias.
            if topk_neg > 0 and topk_neg < split_idx:
                neg_logits = logits[:, :, :split_idx]
                k = min(topk_neg, neg_logits.shape[-1])
                _, idx = neg_logits.topk(k, dim=-1)
                keep = torch.full_like(neg_logits, -1e4)
                keep.scatter_(-1, idx, 0.0)
                logits = torch.cat([neg_logits + keep, logits[:, :, split_idx:]], dim=-1)
            if topk_pos > 0 and topk_pos < (logits.shape[-1] - split_idx):
                pos_logits = logits[:, :, split_idx:]
                k = min(topk_pos, pos_logits.shape[-1])
                _, idx = pos_logits.topk(k, dim=-1)
                keep = torch.full_like(pos_logits, -1e4)
                keep.scatter_(-1, idx, 0.0)
                logits = torch.cat([logits[:, :, :split_idx], pos_logits + keep], dim=-1)
            aff_row = F.softmax(logits, dim=-1)      # normalized over targets
            aff_col = F.softmax(logits, dim=-2)      # normalized over gen
            affinity = torch.sqrt(torch.clamp(aff_row * aff_col, min=1e-6))
            affinity = affinity * targets_w.unsqueeze(1)

            aff_neg = affinity[:, :, :split_idx]   # old_gen + negatives
            aff_pos = affinity[:, :, split_idx:]   # positives

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            r_coeff_pos = aff_pos * sum_neg
            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=-1)  # [B, Cg, T]

            total_force = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)  # should be ~0
            total_force = total_force - total_coeffs.unsqueeze(-1) * old_gen_scaled

            f_norm = (total_force ** 2).mean()
            info[f"loss_R{R}"] = f_norm.detach()
            force_scale = torch.sqrt(torch.clamp(f_norm, min=1e-8))
            force_across_R = force_across_R + total_force / force_scale

        goal_scaled = old_gen_scaled + force_across_R  # detached (no_grad block)

        # diagnostic: what pixel values does the drift force actually demand?
        target_unscaled = goal_scaled * scale_inputs
        info['target_min'] = target_unscaled.min().detach()
        info['target_max'] = target_unscaled.max().detach()
        info['target_mean'] = target_unscaled.mean().detach()
        # quantile has a ~16M-element limit; subsample for diagnostic
        _flat = target_unscaled.flatten()
        if _flat.numel() > 8_000_000:
            _idx = torch.randint(0, _flat.numel(), (8_000_000,), device=_flat.device)
            _flat = _flat[_idx]
        info['target_p99'] = _flat.quantile(0.99).detach()
        info['target_frac_gt_0p5'] = (target_unscaled > 0.5).float().mean().detach()

    gen_scaled = gen / scale_inputs  # scale_inputs is already detached
    diff = gen_scaled - goal_scaled
    loss = diff.pow(2).mean(dim=(-1, -2))  # [B]
    return loss, info
