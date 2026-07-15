from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass(frozen=True)
class LeJEPALoss:
    """The three terms exposed by the LeJEPA objective."""

    total: torch.Tensor
    prediction: torch.Tensor
    sigreg: torch.Tensor


def make_masked_views(
    input_ids: torch.Tensor,
    *,
    num_views: int,
    mask_ratio: float,
    mask_token_id: int,
    seed: int,
) -> list[torch.Tensor]:
    """Create independent text views by masking inputs before the encoder."""
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}")
    if num_views < 2:
        raise ValueError("LeJEPA requires at least two global views")
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be strictly between 0 and 1")
    if input_ids.size(1) < 2:
        raise ValueError("LeJEPA views require sequences of at least two tokens")

    generator = torch.Generator(device=input_ids.device)
    generator.manual_seed(seed % (2**63 - 1))
    views: list[torch.Tensor] = []
    for view_idx in range(num_views):
        mask = torch.rand(input_ids.shape, device=input_ids.device, generator=generator) < mask_ratio
        # Every sample must contain context and corruption. The deterministic
        # fallbacks avoid consuming the process-wide RNG state.
        for sample_idx in range(input_ids.size(0)):
            if not mask[sample_idx].any():
                mask[sample_idx, (sample_idx + view_idx) % input_ids.size(1)] = True
            if mask[sample_idx].all():
                mask[sample_idx, (sample_idx + view_idx + 1) % input_ids.size(1)] = False
        views.append(input_ids.masked_fill(mask, mask_token_id))
    return views


def _distributed_average(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    # torch.distributed.nn.functional provides an autograd-aware collective.
    from torch.distributed.nn.functional import all_reduce

    return all_reduce(value, op=dist.ReduceOp.SUM) / dist.get_world_size()


def sigreg_loss(
    embeddings: torch.Tensor,
    *,
    global_step: int,
    num_slices: int = 256,
    num_knots: int = 17,
    t_max: float = 5.0,
) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization with Epps-Pulley.

    This follows Algorithm 1 of the LeJEPA paper: unit random directions,
    an empirical characteristic function, the N(0, 1) characteristic
    function, Gaussian weighting, and trapezoidal quadrature.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [batch, dimension], got {tuple(embeddings.shape)}")
    if embeddings.size(0) == 0 or embeddings.size(1) == 0:
        raise ValueError("embeddings cannot be empty")
    if num_slices <= 0:
        raise ValueError("num_slices must be positive")
    if num_knots < 2:
        raise ValueError("num_knots must be at least two")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")

    work_dtype = embeddings.dtype
    if work_dtype in {torch.float16, torch.bfloat16}:
        work_dtype = torch.float32
    x = embeddings.to(work_dtype)

    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(global_step % (2**63 - 1))
    directions = torch.randn(
        (x.size(1), num_slices),
        device=x.device,
        dtype=work_dtype,
        generator=generator,
    )
    directions = F.normalize(directions, p=2, dim=0)

    t = torch.linspace(-t_max, t_max, num_knots, device=x.device, dtype=work_dtype)
    normal_cf = torch.exp(-0.5 * t.square())
    phases = (x @ directions).unsqueeze(-1) * t
    ecf_real = _distributed_average(torch.cos(phases).mean(dim=0))
    ecf_imag = _distributed_average(torch.sin(phases).mean(dim=0))

    error = (ecf_real - normal_cf).square() + ecf_imag.square()
    weighted_error = error * normal_cf
    world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    sample_count = embeddings.size(0) * world_size
    directional_statistics = torch.trapezoid(weighted_error, t, dim=-1) * sample_count
    return directional_statistics.mean()


def lejepa_loss(
    global_embeddings: list[torch.Tensor],
    *,
    lambd: float,
    global_step: int,
    num_slices: int = 256,
    num_knots: int = 17,
    t_max: float = 5.0,
    all_embeddings: list[torch.Tensor] | None = None,
) -> LeJEPALoss:
    """Compute Algorithm 2 of LeJEPA without stop-gradient or a teacher."""
    if not 0.0 <= lambd <= 1.0:
        raise ValueError("lambd must be between 0 and 1")
    if len(global_embeddings) < 2:
        raise ValueError("LeJEPA requires at least two global-view embeddings")
    if all_embeddings is None:
        all_embeddings = global_embeddings
    if not all_embeddings:
        raise ValueError("all_embeddings cannot be empty")

    expected_shape = global_embeddings[0].shape
    if len(expected_shape) != 2:
        raise ValueError("each view embedding must have shape [batch, dimension]")
    for embedding in [*global_embeddings, *all_embeddings]:
        if embedding.shape != expected_shape:
            raise ValueError("all view embeddings must have the same shape")

    centers = torch.stack(global_embeddings, dim=0).mean(dim=0)
    stacked_views = torch.stack(all_embeddings, dim=0)
    prediction = (centers.unsqueeze(0) - stacked_views).square().mean()
    sigreg = torch.stack(
        [
            sigreg_loss(
                embedding,
                global_step=global_step,
                num_slices=num_slices,
                num_knots=num_knots,
                t_max=t_max,
            )
            for embedding in all_embeddings
        ]
    ).mean()
    total = (1.0 - lambd) * prediction + lambd * sigreg
    return LeJEPALoss(total=total, prediction=prediction, sigreg=sigreg)
