# edm_diffusers
EDM (Elucidated Diffusion Models) implemented with diffusers modules and PyTorch.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Train

```bash
python scripts/train.py --dataset cifar10 --data-dir data --epochs 20 --batch-size 64
```

Use `--download` to fetch CIFAR-10 if it is not already present.

## YAML config

Use `scripts/run.py` with a YAML config for local folders or Hugging Face datasets, plus preprocessing,
EMA, prediction type, and sampling.

```bash
python scripts/run.py --config configs/example.yaml
```

To run inference:

```bash
python scripts/run.py --config configs/example.yaml --mode infer
```

Key options:
- `dataset.type`: `local` (folder of png/jpg) or `hf` (Hugging Face dataset name).
- `preprocess.resize`: keep aspect ratio on the short edge, then `preprocess.crop` (`center` or `random`).
- `training.predict_type`: `data`, `noise`, or `velocity`.
- `sampling.method`: `heun`, `euler`, or `ddim`.

## Sample

```bash
python scripts/sample.py --checkpoint runs/checkpoint_0001000.pt --batch-size 16 --out samples.png
```
