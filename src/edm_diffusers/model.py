"""Model builders using diffusers UNet2DModel."""

from __future__ import annotations

import unittest
from typing import Iterable, Sequence
import torch
from diffusers import UNet2DModel


def _make_block_types(
    image_size: int,
    block_out_channels: Sequence[int],
    attention_resolutions: Iterable[int],
) -> tuple[list[str], list[str]]:
    attention_set = set(attention_resolutions)
    down = []
    resolution = image_size
    for _ in block_out_channels:
        if resolution in attention_set:
            down.append("AttnDownBlock2D")
        else:
            down.append("DownBlock2D")
        resolution //= 2

    up = []
    for block in reversed(down):
        if block == "AttnDownBlock2D":
            up.append("AttnUpBlock2D")
        else:
            up.append("UpBlock2D")

    return down, up


def create_unet(
    image_size: int = 32,
    in_channels: int = 3,
    out_channels: int = 3,
    base_channels: int = 64,
    channel_mults: Sequence[int] = (1, 2, 2, 2),
    attention_resolutions: Iterable[int] = (16,),
    layers_per_block: int = 2,
    dropout: float = 0.0,
) -> UNet2DModel:
    """Create a UNet backbone suitable for EDM training."""
    block_out_channels = tuple(base_channels * mult for mult in channel_mults)
    down_block_types, up_block_types = _make_block_types(
        image_size=image_size,
        block_out_channels=block_out_channels,
        attention_resolutions=attention_resolutions,
    )

    return UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        dropout=dropout,
    )


def _count_layers(model: torch.nn.Module) -> int:
    """Count the total number of layers (modules) in the model."""
    return len(list(model.modules()))


def _count_long_skips(model: UNet2DModel) -> int:
    """Count the number of long skip connections in the UNet.
    
    In a UNet architecture, long skips connect corresponding down and up blocks.
    The number of skip connections equals the number of down blocks.
    """
    return len(model.down_blocks)


class TestModel(unittest.TestCase):
    def test_model_stats(self):
        """Test that prints model statistics: trainable weights, layers, and long skips."""
        model = create_unet(
            image_size=256,
            in_channels=3,
            out_channels=3,
            base_channels=128,
            channel_mults=(1, 1, 2, 2, 4, 4),
            attention_resolutions=(16,),
            layers_per_block=2,
            dropout=0.0,
        )
        
        # Count trainable parameters
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Count layers
        num_layers = _count_layers(model)
        
        # Count long skips
        num_long_skips = _count_long_skips(model)
        
        # Print statistics
        print(f"\nModel Statistics:")
        print(f"  Total trainable weights: {num_trainable:,}")
        print(f"  Number of layers: {num_layers}")
        print(f"  Number of long skips: {num_long_skips}")


if __name__ == "__main__":
    unittest.main()
