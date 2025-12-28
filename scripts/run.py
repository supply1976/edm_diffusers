#!/usr/bin/env python
"""Run training or sampling from a YAML config."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from edm_diffusers import EMA, create_unet, edm_loss, edm_sampler
from edm_diffusers.data import PreprocessConfig, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDM training/inference from YAML config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=["train", "infer"], default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_device(config: dict) -> torch.device:
    device_name = config.get("device")
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_method(method: str) -> str:
    method = method.lower()
    if method in {"edm_heun", "heun"}:
        return "heun"
    if method in {"edm_euler", "euler"}:
        return "euler"
    if method == "ddim":
        return "ddim"
    return method


def resolve_paths(config: dict, base_dir: Path) -> None:
    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("type", "local") == "local" and "path" in dataset_cfg:
        path = Path(dataset_cfg["path"])
        if not path.is_absolute():
            dataset_cfg["path"] = str((base_dir / path).resolve())

    training_cfg = config.get("training", {})
    if "output_dir" in training_cfg:
        output_dir = Path(training_cfg["output_dir"])
        if not output_dir.is_absolute():
            training_cfg["output_dir"] = str((base_dir / output_dir).resolve())

    inference_cfg = config.get("inference", {})
    for key in ("checkpoint", "output"):
        if key in inference_cfg:
            path = Path(inference_cfg[key])
            if not path.is_absolute():
                inference_cfg[key] = str((base_dir / path).resolve())


def train_from_config(config: dict) -> None:
    seed = config.get("seed", 42)
    torch.manual_seed(seed)

    device = select_device(config)
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    dataset_config = config.get("dataset", {})

    image_size = model_config.get("image_size", 32)
    channels = model_config.get("in_channels", model_config.get("channels", 3))
    model_config.setdefault("image_size", image_size)
    model_config.setdefault("in_channels", channels)
    model_config.setdefault("out_channels", channels)

    preprocess_cfg = config.get("preprocess", {})
    preprocess = PreprocessConfig(
        resize=preprocess_cfg.get("resize"),
        crop=preprocess_cfg.get("crop", "center"),
        crop_size=preprocess_cfg.get("crop_size"),
    )

    dataset = build_dataset(dataset_config, preprocess, image_size=image_size, channels=channels)

    batch_size = training_config.get("batch_size", 64)
    num_workers = training_config.get("num_workers", 4)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_unet(**model_config).to(device)
    learning_rate = training_config.get("learning_rate", training_config.get("lr", 2e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    predict_type = training_config.get("predict_type", "data")
    sigma_min = training_config.get("sigma_min", 0.002)
    sigma_max = training_config.get("sigma_max", 80.0)
    sigma_data = training_config.get("sigma_data", 0.5)

    ema_cfg = training_config.get("ema", {})
    ema = None
    if ema_cfg.get("enabled", False):
        ema = EMA(model, decay=ema_cfg.get("decay", 0.999))

    output_dir = Path(training_config.get("output_dir", "runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    epochs = training_config.get("epochs", 20)
    max_steps = training_config.get("iterations", training_config.get("max_steps", 0))
    save_every = training_config.get("save_every", 1000)
    log_every = training_config.get("log_every", 100)
    ema_update_every = max(1, ema_cfg.get("update_every", 1))

    def save_checkpoint(step: int) -> None:
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": model_config,
            "training_config": {
                "predict_type": predict_type,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "sigma_data": sigma_data,
            },
        }
        if ema is not None:
            checkpoint["ema"] = ema.state_dict()
        torch.save(checkpoint, output_dir / f"checkpoint_{step:07d}.pt")

    step = 0
    model.train()
    progress = tqdm(total=max_steps if max_steps else None)

    for _epoch in range(epochs):
        for images, _ in loader:
            images = images.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = edm_loss(
                model,
                images,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sigma_data=sigma_data,
                prediction_type=predict_type,
            )
            loss.backward()
            optimizer.step()

            step += 1
            if ema is not None and step % ema_update_every == 0:
                ema.update(model)

            progress.update(1)
            if step % log_every == 0:
                progress.set_description(f"loss={loss.item():.4f}")

            if step % save_every == 0:
                save_checkpoint(step)

            if max_steps and step >= max_steps:
                break

        if max_steps and step >= max_steps:
            break

    save_checkpoint(step)
    progress.close()


def infer_from_config(config: dict) -> None:
    device = select_device(config)
    inference_cfg = config.get("inference", {})
    sampling_cfg = config.get("sampling", {})

    checkpoint_path = Path(inference_cfg["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_config = checkpoint.get("model_config", config.get("model"))
    if not model_config:
        raise ValueError("Missing model config in checkpoint and YAML")

    model = create_unet(**model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if inference_cfg.get("use_ema", False) and "ema" in checkpoint:
        ema = EMA.from_state_dict(model, checkpoint["ema"])
        ema.apply_to(model)

    training_cfg = checkpoint.get("training_config", {})
    predict_type = (
        inference_cfg.get("predict_type")
        or sampling_cfg.get("predict_type")
        or training_cfg.get("predict_type")
        or "data"
    )

    sigma_min = sampling_cfg.get("sigma_min", training_cfg.get("sigma_min", 0.002))
    sigma_max = sampling_cfg.get("sigma_max", training_cfg.get("sigma_max", 80.0))
    sigma_data = sampling_cfg.get("sigma_data", training_cfg.get("sigma_data", 0.5))
    rho = sampling_cfg.get("rho", 7.0)

    steps = sampling_cfg.get("steps", 40)
    method = parse_method(sampling_cfg.get("method", "heun"))

    batch_size = inference_cfg.get("batch_size", 16)
    seed = inference_cfg.get("seed")

    samples = edm_sampler(
        model,
        shape=(
            batch_size,
            model_config["out_channels"],
            model_config["image_size"],
            model_config["image_size"],
        ),
        num_steps=steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        sigma_data=sigma_data,
        device=device,
        seed=seed,
        prediction_type=predict_type,
        method=method,
    )

    samples = (samples.clamp(-1, 1) + 1) / 2
    output_path = Path(inference_cfg.get("output", "samples.png"))
    nrow = int(sqrt(batch_size)) or 1
    save_image(samples, output_path, nrow=nrow)
    print(f"Saved samples to {output_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    resolve_paths(config, args.config.parent)

    mode = args.mode or config.get("mode", "train")
    if mode == "train":
        train_from_config(config)
    elif mode == "infer":
        infer_from_config(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
