"""Exponential moving average helper."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EMAState:
    decay: float
    shadow: dict[str, torch.Tensor]


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = param.detach().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    @classmethod
    def from_state_dict(cls, model: torch.nn.Module, state: dict) -> "EMA":
        ema = cls(model, decay=state.get("decay", 0.999))
        ema.shadow = {k: v.clone() for k, v in state.get("shadow", {}).items()}
        return ema
