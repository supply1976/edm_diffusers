"""Core EDM utilities (loss, preconditioning, sampler)."""

from __future__ import annotations

import math
from typing import Iterable

import torch


def sample_sigmas(
    batch_size: int,
    sigma_min: float,
    sigma_max: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample log-uniform noise scales for EDM training."""
    log_sigma_min = math.log(sigma_min)
    log_sigma_max = math.log(sigma_max)
    u = torch.rand(batch_size, device=device, dtype=dtype)
    return torch.exp(log_sigma_min + (log_sigma_max - log_sigma_min) * u)


def _normalize_prediction_type(prediction_type: str | None) -> str:
    if prediction_type is None:
        return "data"
    pred = prediction_type.lower()
    if pred in {"data", "x0", "sample"}:
        return "data"
    if pred in {"noise", "epsilon", "eps"}:
        return "noise"
    if pred in {"velocity", "v", "v_prediction"}:
        return "velocity"
    raise ValueError(f"Unknown prediction type: {prediction_type}")


def _model_forward(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float,
) -> torch.Tensor:
    sigma_data_t = torch.tensor(sigma_data, device=x.device, dtype=x.dtype)
    sigma = sigma.view(-1, 1, 1, 1)

    sigma_sq = sigma * sigma
    sigma_data_sq = sigma_data_t * sigma_data_t

    c_in = 1.0 / torch.sqrt(sigma_sq + sigma_data_sq)
    c_noise = torch.log(sigma.view(-1)) / 4.0
    return model(c_in * x, c_noise).sample


def preconditioned_denoiser(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
    prediction_type: str | None = "data",
) -> torch.Tensor:
    """Apply EDM preconditioning around a diffusers UNet2DModel."""
    pred = _normalize_prediction_type(prediction_type)
    sigma_data_t = torch.tensor(sigma_data, device=x.device, dtype=x.dtype)
    sigma = sigma.view(-1, 1, 1, 1)

    sigma_sq = sigma * sigma
    sigma_data_sq = sigma_data_t * sigma_data_t
    model_out = _model_forward(model, x, sigma, sigma_data)

    if pred == "data":
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * sigma_data_t / torch.sqrt(sigma_sq + sigma_data_sq)
        return c_skip * x + c_out * model_out

    if pred == "noise":
        return x - sigma * model_out

    if pred == "velocity":
        return (x - sigma * model_out) / (1.0 + sigma_sq)

    raise ValueError(f"Unknown prediction type: {prediction_type}")


def edm_loss(
    model,
    clean_images: torch.Tensor,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    sigma_data: float = 0.5,
    prediction_type: str | None = "data",
) -> torch.Tensor:
    """Compute EDM loss with noise conditioning and preconditioning."""
    pred = _normalize_prediction_type(prediction_type)
    batch_size = clean_images.shape[0]
    sigmas = sample_sigmas(
        batch_size,
        sigma_min,
        sigma_max,
        device=clean_images.device,
        dtype=clean_images.dtype,
    )

    noise = torch.randn_like(clean_images)
    sigma_view = sigmas.view(-1, 1, 1, 1)
    noised = clean_images + sigma_view * noise

    if pred == "data":
        denoised = preconditioned_denoiser(
            model, noised, sigmas, sigma_data, prediction_type=pred
        )
        sigma_data_t = torch.tensor(
            sigma_data, device=clean_images.device, dtype=clean_images.dtype
        )
        weight = (sigmas * sigmas + sigma_data_t * sigma_data_t) / (
            (sigmas * sigma_data_t) ** 2
        )
        weight = weight.view(-1, 1, 1, 1)
        loss = weight * (denoised - clean_images).pow(2)
        return loss.mean()

    model_out = _model_forward(model, noised, sigmas, sigma_data)
    if pred == "noise":
        target = noise
    elif pred == "velocity":
        target = noise - sigma_view * clean_images
    else:
        raise ValueError(f"Unknown prediction type: {prediction_type}")

    loss = (model_out - target).pow(2)
    return loss.mean()


def karras_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Noise schedule from the EDM paper (Karras et al.)."""
    ramp = torch.linspace(0, 1, num_steps, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


def _append_zero(sigmas: torch.Tensor) -> torch.Tensor:
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def edm_sampler(
    model,
    shape: Iterable[int],
    num_steps: int = 40,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    sigma_data: float = 0.5,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int | None = None,
    prediction_type: str | None = "data",
    method: str = "heun",
) -> torch.Tensor:
    """Sample with EDM solvers (Heun, Euler, or DDIM-style)."""
    batch_size = int(shape[0])
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    sigmas = karras_schedule(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        device=device,
        dtype=dtype,
    )
    sigmas = _append_zero(sigmas)

    x = torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator) * sigma_max

    pred = _normalize_prediction_type(prediction_type)
    method = method.lower()

    model.eval()
    with torch.no_grad():
        for i in range(num_steps):
            sigma = sigmas[i]
            sigma_next = sigmas[i + 1]

            sigma_vec = torch.full((batch_size,), float(sigma), device=device, dtype=dtype)
            sigma_view = sigma_vec.view(-1, 1, 1, 1)
            denoised = preconditioned_denoiser(
                model, x, sigma_vec, sigma_data, prediction_type=pred
            )

            if method == "ddim":
                eps = (x - denoised) / sigma_view
                x = denoised + eps * sigma_next
                continue

            d = (x - denoised) / sigma_view
            dt = sigma_next - sigma

            if method == "euler":
                x = x + d * dt
                continue

            if method != "heun":
                raise ValueError(f"Unknown sampling method: {method}")

            x_euler = x + d * dt
            if float(sigma_next) > 0:
                sigma_next_vec = torch.full(
                    (batch_size,), float(sigma_next), device=device, dtype=dtype
                )
                sigma_next_view = sigma_next_vec.view(-1, 1, 1, 1)
                denoised_next = preconditioned_denoiser(
                    model, x_euler, sigma_next_vec, sigma_data, prediction_type=pred
                )
                d_next = (x_euler - denoised_next) / sigma_next_view
                x = x + 0.5 * (d + d_next) * dt
            else:
                x = x_euler

    return x
