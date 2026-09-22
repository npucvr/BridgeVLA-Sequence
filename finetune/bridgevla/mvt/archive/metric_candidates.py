"""Archived drift-metric candidates removed from the runtime filter.

The current runtime keeps only ``diagonal_robust_mahalanobis`` in
``mvt.filt3r_akf``.  These functions preserve the previous candidate metrics
for historical comparison or a future explicitly authorized ablation; this
module is intentionally not imported by the runtime filter.
"""

# gbw____
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _signed_simplex_probability(
    value: torch.Tensor,
    eps: float,
    channel_groups: Optional[int] = None,
) -> torch.Tensor:
    channels = value.shape[-1]
    groups = channel_groups
    if groups is not None and (groups <= 0 or channels % groups != 0):
        groups = None
    positive = F.softplus(value)
    negative = F.softplus(-value)
    if groups is not None:
        width = channels // groups
        positive = positive.reshape(*positive.shape[:-1], groups, width).sum(-1)
        negative = negative.reshape(*negative.shape[:-1], groups, width).sum(-1)
    mass = torch.cat((positive, negative), dim=-1).clamp_min(eps)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(eps)


def _js_probability_distance(
    left_prob: torch.Tensor,
    right_prob: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    midpoint = 0.5 * (left_prob + right_prob)
    left_kl = left_prob * (left_prob.log() - midpoint.clamp_min(eps).log())
    right_kl = right_prob * (right_prob.log() - midpoint.clamp_min(eps).log())
    js = 0.5 * (left_kl.sum(dim=-1) + right_kl.sum(dim=-1))
    return torch.sqrt((js / math.log(2.0)).clamp_min(0.0))


def _signed_js_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    eps: float,
    channel_groups: Optional[int] = None,
) -> torch.Tensor:
    left_prob = _signed_simplex_probability(left, eps, channel_groups)
    right_prob = _signed_simplex_probability(right, eps, channel_groups)
    return _js_probability_distance(left_prob, right_prob, eps)


def _local_fused_ot_js(
    previous: torch.Tensor,
    current: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if previous.ndim != 5 or current.ndim != 5:
        return _signed_js_distance(previous, current, eps)
    batch, views, height, width, channels = current.shape
    if height < 2 or width < 2:
        return _signed_js_distance(previous, current, eps)
    grid_shape = (batch * views, channels, height, width)
    previous_grid = previous.permute(0, 1, 4, 2, 3).reshape(grid_shape)
    current_grid = current.permute(0, 1, 4, 2, 3).reshape(grid_shape)
    previous_patch = F.unfold(
        previous_grid, kernel_size=3, padding=1
    ).transpose(1, 2).reshape(batch * views * height * width, 9, channels)
    current_patch = F.unfold(
        current_grid, kernel_size=3, padding=1
    ).transpose(1, 2).reshape(batch * views * height * width, 9, channels)

    groups = 32 if channels % 32 == 0 else None
    previous_prob = _signed_simplex_probability(previous_patch, eps, groups)
    current_prob = _signed_simplex_probability(current_patch, eps, groups)
    sample_count = current_prob.shape[0]
    feature_cost = []
    for key in range(9):
        feature_cost.append(
            _js_probability_distance(
                current_prob,
                previous_prob[:, key, :].unsqueeze(1),
                eps,
            )
        )
    feature_cost = torch.stack(feature_cost, dim=-1)
    offsets = torch.tensor((-1, 0, 1), device=current.device, dtype=current.dtype)
    offset_rows = torch.repeat_interleave(offsets, 3)
    offset_cols = offsets.repeat(3)
    position_cost = (
        (offset_rows[:, None] - offset_rows[None, :]).square()
        + (offset_cols[:, None] - offset_cols[None, :]).square()
    ) / 8.0
    cost = 0.75 * feature_cost + 0.25 * position_cost.unsqueeze(0)
    mass_log = math.log(1.0 / 9.0)
    log_kernel = -cost / 0.10
    left_potential = torch.zeros(
        sample_count, 9, device=current.device, dtype=current.dtype
    )
    right_potential = torch.zeros_like(left_potential)
    for _ in range(6):
        left_potential = mass_log - torch.logsumexp(
            log_kernel + right_potential.unsqueeze(1), dim=-1
        )
        right_potential = mass_log - torch.logsumexp(
            log_kernel + left_potential.unsqueeze(-1), dim=-2
        )
    plan = torch.exp(
        log_kernel
        + left_potential.unsqueeze(-1)
        + right_potential.unsqueeze(1)
    )
    row_cost = (plan[:, 4, :] * cost[:, 4, :]).sum(dim=-1) * 9.0
    return row_cost.reshape(batch, views, height, width).clamp_min(0.0)


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    root = eigenvalues.clamp_min(0.0).sqrt()
    return eigenvectors @ torch.diag_embed(root) @ eigenvectors.transpose(-1, -2)


def _bures_lowrank_local(
    previous: torch.Tensor,
    current: torch.Tensor,
    eps: float,
    delta_floor: float,
) -> torch.Tensor:
    if previous.ndim != 5 or current.ndim != 5:
        return _signed_js_distance(previous, current, eps)
    batch, views, height, width, channels = current.shape
    groups = 4 if channels % 4 == 0 else None
    if groups is None or height < 2 or width < 2:
        return _signed_js_distance(previous, current, eps)
    grid_shape = (batch * views, channels, height, width)
    previous_grid = previous.permute(0, 1, 4, 2, 3).reshape(grid_shape)
    current_grid = current.permute(0, 1, 4, 2, 3).reshape(grid_shape)
    previous_patch = F.unfold(
        previous_grid, kernel_size=3, padding=1
    ).transpose(1, 2).reshape(batch * views * height * width, 9, channels)
    current_patch = F.unfold(
        current_grid, kernel_size=3, padding=1
    ).transpose(1, 2).reshape(batch * views * height * width, 9, channels)
    group_width = channels // groups
    previous_features = previous_patch.reshape(-1, 9, groups, group_width).mean(-1)
    current_features = current_patch.reshape(-1, 9, groups, group_width).mean(-1)
    previous_mean = previous_features.mean(dim=1)
    current_mean = current_features.mean(dim=1)
    previous_centered = previous_features - previous_mean.unsqueeze(1)
    current_centered = current_features - current_mean.unsqueeze(1)
    denominator = float(max(1, previous_features.shape[1] - 1))
    previous_covariance = torch.einsum(
        "nsi,nsj->nij", previous_centered, previous_centered
    ) / denominator
    current_covariance = torch.einsum(
        "nsi,nsj->nij", current_centered, current_centered
    ) / denominator
    floor = delta_floor * torch.eye(
        groups, device=current.device, dtype=current.dtype
    )
    previous_covariance = previous_covariance + floor
    current_covariance = current_covariance + floor
    current_root = _matrix_sqrt_psd(current_covariance)
    middle = current_root @ previous_covariance @ current_root
    middle_root = _matrix_sqrt_psd(middle)
    covariance_term = torch.diagonal(
        current_covariance + previous_covariance - 2.0 * middle_root,
        dim1=-2,
        dim2=-1,
    ).sum(-1).clamp_min(0.0) / float(groups)
    mean_term = (current_mean - previous_mean).square().mean(-1)
    score = torch.sqrt((mean_term + covariance_term).clamp_min(0.0))
    return score.reshape(batch, views, height, width)


def pairwise_drift_metric(
    previous: torch.Tensor,
    current: torch.Tensor,
    previous_step: Optional[torch.Tensor],
    metric: str,
    eps: float,
    delta_floor: float,
) -> torch.Tensor:
    """Reproduce the pre-archive candidate metric dispatcher."""

    step = current - previous
    rms_l2 = torch.sqrt(step.square().mean(dim=-1).clamp_min(0.0))
    if metric == "rms_l2":
        return rms_l2

    previous_rms = torch.sqrt(previous.square().mean(dim=-1).clamp_min(eps))
    if metric == "relative_l2":
        return rms_l2 / (previous_rms + eps)
    if metric == "l1_mean":
        return step.abs().mean(dim=-1) / (previous_rms + eps)
    if metric == "fractional_l05":
        fractional_mean = step.abs().clamp_min(0.0).sqrt().mean(dim=-1)
        return fractional_mean.square() / (previous_rms + eps)
    if metric in {
        "relative_topk_k4",
        "relative_topk_k16",
        "relative_max",
        "relative_exceedance",
    }:
        relative_abs = step.abs() / (previous_rms.unsqueeze(-1) + eps)
        if metric == "relative_max":
            return relative_abs.max(dim=-1).values
        if metric == "relative_exceedance":
            exceedance = (relative_abs - 3.0).clamp_min(0.0)
            return torch.sqrt(exceedance.square().mean(dim=-1).clamp_min(0.0))
        k = 4 if metric == "relative_topk_k4" else 16
        topk = torch.topk(
            relative_abs, k=min(k, relative_abs.shape[-1]), dim=-1
        ).values
        return torch.sqrt(topk.square().mean(dim=-1).clamp_min(0.0))
    if metric == "js_signed_simplex":
        js_distance = _signed_js_distance(previous, current, eps)
        radial = torch.log(
            (torch.linalg.vector_norm(current, dim=-1) + eps)
            / (torch.linalg.vector_norm(previous, dim=-1) + eps)
        ).abs()
        return torch.sqrt((js_distance.square() + radial.square()).clamp_min(0.0))
    if metric == "local_fused_ot_js":
        return _local_fused_ot_js(previous, current, eps)
    if metric == "bures_lowrank_local":
        return _bures_lowrank_local(previous, current, eps, delta_floor)
    if metric == "radial_angular":
        current_rms = torch.sqrt(current.square().mean(dim=-1).clamp_min(eps))
        previous_norm = torch.linalg.vector_norm(previous, dim=-1)
        current_norm = torch.linalg.vector_norm(current, dim=-1)
        norm_product = current_norm * previous_norm
        cosine = torch.where(
            norm_product > eps,
            torch.sum(current * previous, dim=-1) / (norm_product + eps),
            torch.ones_like(norm_product),
        ).clamp(min=-1.0, max=1.0)
        radial = torch.log((current_rms + eps) / (previous_rms + eps)).abs()
        angular = torch.sqrt((2.0 * (1.0 - cosine)).clamp_min(0.0))
        return torch.sqrt(radial.square() + angular.square())
    if metric == "angular_geodesic":
        previous_norm = torch.linalg.vector_norm(previous, dim=-1)
        current_norm = torch.linalg.vector_norm(current, dim=-1)
        norm_product = previous_norm * current_norm
        cosine = torch.where(
            norm_product > eps,
            torch.sum(previous * current, dim=-1) / (norm_product + eps),
            torch.ones_like(norm_product),
        ).clamp(min=-1.0, max=1.0)
        return torch.sqrt((2.0 * (1.0 - cosine)).clamp_min(0.0))
    if metric == "centered_correlation":
        previous_centered = previous - previous.mean(dim=-1, keepdim=True)
        current_centered = current - current.mean(dim=-1, keepdim=True)
        previous_norm = torch.linalg.vector_norm(previous_centered, dim=-1)
        current_norm = torch.linalg.vector_norm(current_centered, dim=-1)
        norm_product = previous_norm * current_norm
        correlation = torch.where(
            norm_product > eps,
            torch.sum(previous_centered * current_centered, dim=-1)
            / (norm_product + eps),
            torch.ones_like(norm_product),
        ).clamp(min=-1.0, max=1.0)
        return torch.sqrt((2.0 * (1.0 - correlation)).clamp_min(0.0))
    if metric == "huber_channel":
        previous_rms = torch.sqrt(
            previous.square().mean(dim=-1, keepdim=True).clamp_min(eps)
        )
        robust_scale = previous.abs() + 0.25 * previous_rms + eps
        standardized = (step / robust_scale).clamp(min=-8.0, max=8.0)
        absolute = standardized.abs()
        kappa = 1.5
        loss = torch.where(
            absolute <= kappa,
            0.5 * standardized.square(),
            kappa * (absolute - 0.5 * kappa),
        )
        return torch.sqrt((2.0 * loss).mean(dim=-1).clamp_min(0.0))
    if metric == "robust_l2":
        absolute_step = step.abs()
        center = torch.median(absolute_step, dim=-1, keepdim=True).values
        mad = torch.median(
            (absolute_step - center).abs(), dim=-1, keepdim=True
        ).values
        upper = center + 3.0 * 1.4826 * mad + eps
        clipped = torch.minimum(absolute_step, upper)
        return torch.sqrt(clipped.square().mean(dim=-1).clamp_min(0.0))
    if metric == "relative_l2_cosine":
        relative = rms_l2 / (previous_rms + eps)
        if previous_step is None:
            return relative
        current_step_norm = torch.linalg.vector_norm(step, dim=-1)
        previous_step_norm = torch.linalg.vector_norm(previous_step, dim=-1)
        step_product = current_step_norm * previous_step_norm
        direction_cosine = torch.where(
            step_product > eps,
            torch.sum(step * previous_step, dim=-1) / (step_product + eps),
            torch.ones_like(step_product),
        ).clamp(min=0.0, max=1.0)
        return relative * (0.5 + 0.5 * direction_cosine)
    raise ValueError(f"archived metric is not recognized: {metric}")
# ____
