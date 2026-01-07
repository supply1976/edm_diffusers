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


def _extract_attention_heads(block) -> int | None:
    """Extract number of attention heads from a block if it has attention."""
    if hasattr(block, 'attentions') and block.attentions:
        for attn_block in block.attentions:
            if hasattr(attn_block, 'transformer_blocks'):
                for transformer_block in attn_block.transformer_blocks:
                    if hasattr(transformer_block, 'attn1') and hasattr(transformer_block.attn1, 'heads'):
                        return transformer_block.attn1.heads
            # Check for direct Attention module
            if hasattr(attn_block, 'heads'):
                return attn_block.heads
            # Check for group_norm and query
            if hasattr(attn_block, 'query'):
                query = attn_block.query
                if hasattr(query, 'out_features') and hasattr(attn_block, 'head_dim'):
                    return query.out_features // attn_block.head_dim
    return None


def _get_model_architecture_info(model: UNet2DModel) -> dict:
    """Extract detailed architecture information from the model."""
    info = {
        'config': {},
        'blocks': {
            'down_blocks': [],
            'mid_block': {},
            'up_blocks': []
        }
    }
    
    # Extract config information
    config = model.config
    info['config']['time_embedding_type'] = config.get('time_embedding_type', 'positional')
    info['config']['time_embedding_dim'] = config.get('time_embedding_dim', None)
    info['config']['attention_head_dim'] = config.get('attention_head_dim', 8)
    info['config']['upsample_type'] = config.get('upsample_type', 'conv')
    info['config']['downsample_type'] = config.get('downsample_type', 'conv')
    info['config']['resnet_time_scale_shift'] = config.get('resnet_time_scale_shift', 'default')
    info['config']['act_fn'] = config.get('act_fn', 'silu')
    info['config']['norm_num_groups'] = config.get('norm_num_groups', 32)
    
    # Extract down blocks info
    for i, block in enumerate(model.down_blocks):
        block_info = {
            'index': i,
            'type': type(block).__name__,
            'has_attention': 'Attn' in type(block).__name__,
            'num_resnets': len(block.resnets) if hasattr(block, 'resnets') else 0,
            'has_downsample': hasattr(block, 'downsamplers') and block.downsamplers is not None,
        }
        
        if block_info['has_attention']:
            num_heads = _extract_attention_heads(block)
            block_info['attention_heads'] = num_heads
        
        info['blocks']['down_blocks'].append(block_info)
    
    # Extract mid block info
    mid = model.mid_block
    info['blocks']['mid_block'] = {
        'type': type(mid).__name__,
        'has_attention': hasattr(mid, 'attentions') and mid.attentions is not None,
        'num_resnets': len(mid.resnets) if hasattr(mid, 'resnets') else 0,
    }
    if info['blocks']['mid_block']['has_attention']:
        num_heads = _extract_attention_heads(mid)
        info['blocks']['mid_block']['attention_heads'] = num_heads
    
    # Extract up blocks info
    for i, block in enumerate(model.up_blocks):
        block_info = {
            'index': i,
            'type': type(block).__name__,
            'has_attention': 'Attn' in type(block).__name__,
            'num_resnets': len(block.resnets) if hasattr(block, 'resnets') else 0,
            'has_upsample': hasattr(block, 'upsamplers') and block.upsamplers is not None,
        }
        
        if block_info['has_attention']:
            num_heads = _extract_attention_heads(block)
            block_info['attention_heads'] = num_heads
        
        info['blocks']['up_blocks'].append(block_info)
    
    return info

class TestModel(unittest.TestCase):
    def test_model_stats(self):
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
        
        # Get detailed architecture info
        arch_info = _get_model_architecture_info(model)
        
        # Print statistics
        print(f"\n{'='*60}")
        print(f"Model Statistics:")
        print(f"{'='*60}")
        print(f"  Total trainable weights: {num_trainable:,}")
        print(f"  Number of layers: {num_layers}")
        
        # Print configuration details
        print(f"\n{'='*60}")
        print(f"Model Configuration:")
        print(f"{'='*60}")
        print(f"  Time Embedding Type: {arch_info['config']['time_embedding_type']}")
        print(f"  Time Embedding Dim: {arch_info['config']['time_embedding_dim']}")
        print(f"  Attention Head Dim: {arch_info['config']['attention_head_dim']}")
        print(f"  Activation Function: {arch_info['config']['act_fn']}")
        print(f"  Normalization Groups: {arch_info['config']['norm_num_groups']}")
        print(f"  Upsampling Type: {arch_info['config']['upsample_type']}")
        print(f"  Downsampling Type: {arch_info['config']['downsample_type']}")
        print(f"  ResNet Time Scale Shift: {arch_info['config']['resnet_time_scale_shift']}")
        
        # Print down blocks architecture
        print(f"\n{'='*60}")
        print(f"Down Blocks Architecture:")
        print(f"{'='*60}")
        for block_info in arch_info['blocks']['down_blocks']:
            print(f"  Block {block_info['index']}: {block_info['type']}")
            print(f"    ResNet Layers: {block_info['num_resnets']}")
            print(f"    Has Attention: {block_info['has_attention']}")
            if block_info.get('attention_heads'):
                print(f"    Attention Heads: {block_info['attention_heads']}")
            print(f"    Has Downsampler: {block_info['has_downsample']}")
        
        # Print mid block architecture
        print(f"\n{'='*60}")
        print(f"Mid Block Architecture:")
        print(f"{'='*60}")
        mid_info = arch_info['blocks']['mid_block']
        print(f"  Type: {mid_info['type']}")
        print(f"  ResNet Layers: {mid_info['num_resnets']}")
        print(f"  Has Attention: {mid_info['has_attention']}")
        if mid_info.get('attention_heads'):
            print(f"  Attention Heads: {mid_info['attention_heads']}")
        
        # Print up blocks architecture
        print(f"\n{'='*60}")
        print(f"Up Blocks Architecture:")
        print(f"{'='*60}")
        for block_info in arch_info['blocks']['up_blocks']:
            print(f"  Block {block_info['index']}: {block_info['type']}")
            print(f"    ResNet Layers: {block_info['num_resnets']}")
            print(f"    Has Attention: {block_info['has_attention']}")
            if block_info.get('attention_heads'):
                print(f"    Attention Heads: {block_info['attention_heads']}")
            print(f"    Has Upsampler: {block_info['has_upsample']}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    unittest.main()
