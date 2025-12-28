#!/usr/bin/env python
"""Sample images from an EDM checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torchvision.utils import save_image

from edm_diffusers import EMA, create_unet, edm_sampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from an EDM checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sigma-min", type=float, default=None)
    parser.add_argument("--sigma-max", type=float, default=None)
    parser.add_argument("--sigma-data", type=float, default=None)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--method", type=str, default="heun")
    parser.add_argument(
        "--predict-type",
        choices=["data", "noise", "velocity"],
        default=None,
    )
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--channel-mults", type=str, default=None)
    parser.add_argument("--attention-resolutions", type=str, default=None)
    parser.add_argument("--layers-per-block", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--out", type=Path, default=Path("samples.png"))
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def parse_method(method: str) -> str:
    method = method.lower()
    if method in {"edm_heun", "heun"}:
        return "heun"
    if method in {"edm_euler", "euler"}:
        return "euler"
    if method == "ddim":
        return "ddim"
    return method


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get("model_config")
    training_config = checkpoint.get("training_config", {})

    if model_config is None:
        if args.image_size is None or args.channels is None:
            raise ValueError("Missing model config. Provide --image-size and --channels.")
        model_config = {
            "image_size": args.image_size,
            "in_channels": args.channels,
            "out_channels": args.channels,
            "base_channels": args.base_channels or 64,
            "channel_mults": parse_int_list(args.channel_mults or "1,2,2,2"),
            "attention_resolutions": parse_int_list(args.attention_resolutions or "16"),
            "layers_per_block": args.layers_per_block or 2,
            "dropout": args.dropout or 0.0,
        }
    else:
        overrides = {
            "image_size": args.image_size,
            "in_channels": args.channels,
            "out_channels": args.channels,
            "base_channels": args.base_channels,
            "channel_mults": parse_int_list(args.channel_mults)
            if args.channel_mults
            else None,
            "attention_resolutions": parse_int_list(args.attention_resolutions)
            if args.attention_resolutions
            else None,
            "layers_per_block": args.layers_per_block,
            "dropout": args.dropout,
        }
        for key, value in overrides.items():
            if value is not None:
                model_config[key] = value

    model = create_unet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.use_ema and "ema" in checkpoint:
        ema = EMA.from_state_dict(model, checkpoint["ema"])
        ema.apply_to(model)

    sigma_min = args.sigma_min if args.sigma_min is not None else training_config.get("sigma_min", 0.002)
    sigma_max = args.sigma_max if args.sigma_max is not None else training_config.get("sigma_max", 80.0)
    sigma_data = args.sigma_data if args.sigma_data is not None else training_config.get("sigma_data", 0.5)
    predict_type = args.predict_type or training_config.get("predict_type", "data")
    method = parse_method(args.method)

    samples = edm_sampler(
        model,
        shape=(args.batch_size, model_config["out_channels"], model_config["image_size"], model_config["image_size"]),
        num_steps=args.steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=args.rho,
        sigma_data=sigma_data,
        device=device,
        seed=args.seed,
        prediction_type=predict_type,
        method=method,
    )

    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples, args.out, nrow=int(args.batch_size ** 0.5))
    print(f"Saved samples to {args.out}")


if __name__ == "__main__":
    main()
