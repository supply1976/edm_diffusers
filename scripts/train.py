#!/usr/bin/env python
"""Train an EDM model using diffusers UNet2DModel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from edm_diffusers import EMA, create_unet, edm_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EDM with diffusers UNet2DModel")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=["cifar10", "imagefolder"], default="cifar10")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,2,2")
    parser.add_argument("--attention-resolutions", type=str, default="16")
    parser.add_argument("--layers-per-block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--predict-type",
        choices=["data", "noise", "velocity"],
        default="data",
    )
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-update-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_dataset(args: argparse.Namespace):
    transform = transforms.Compose(
        [
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * args.channels, [0.5] * args.channels),
        ]
    )

    if args.dataset == "cifar10":
        return datasets.CIFAR10(
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=transform,
        )

    if args.dataset == "imagefolder":
        return datasets.ImageFolder(root=args.data_dir, transform=transform)

    raise ValueError(f"Unknown dataset: {args.dataset}")


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_config: dict,
    training_config: dict,
    ema: EMA | None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_config": model_config,
        "training_config": training_config,
    }
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, output_dir / f"checkpoint_{step:07d}.pt")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_mults = parse_int_list(args.channel_mults)
    attention_resolutions = parse_int_list(args.attention_resolutions)

    model_config = {
        "image_size": args.image_size,
        "in_channels": args.channels,
        "out_channels": args.channels,
        "base_channels": args.base_channels,
        "channel_mults": channel_mults,
        "attention_resolutions": attention_resolutions,
        "layers_per_block": args.layers_per_block,
        "dropout": args.dropout,
    }

    model = create_unet(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay) if args.ema else None

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    training_config = {
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "sigma_data": args.sigma_data,
        "dataset": args.dataset,
        "predict_type": args.predict_type,
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({"model": model_config, "training": training_config}, indent=2)
    )

    step = 0
    model.train()
    progress = tqdm(total=args.max_steps if args.max_steps > 0 else None)

    for epoch in range(args.epochs):
        for images, _ in loader:
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = edm_loss(
                model,
                images,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                sigma_data=args.sigma_data,
                prediction_type=args.predict_type,
            )
            loss.backward()
            optimizer.step()
            if ema is not None and step % max(1, args.ema_update_every) == 0:
                ema.update(model)

            step += 1
            progress.update(1)
            if step % args.log_every == 0:
                progress.set_description(f"loss={loss.item():.4f}")

            if step % args.save_every == 0:
                save_checkpoint(
                    output_dir,
                    model,
                    optimizer,
                    step,
                    model_config,
                    training_config,
                    ema,
                )

            if args.max_steps and step >= args.max_steps:
                break

        if args.max_steps and step >= args.max_steps:
            break

    save_checkpoint(output_dir, model, optimizer, step, model_config, training_config, ema)
    progress.close()


if __name__ == "__main__":
    main()
