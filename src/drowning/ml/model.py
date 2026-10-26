"""MobileNetV3-small image encoder with image and causal temporal heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
except ImportError as exc:  # pragma: no cover - depends on optional environment
    raise ImportError("Install the optional ML dependencies with `pip install -e '.[ml]'`.") from exc

CLASS_KEYS = ("drowning", "swimming", "out_of_water")
ONTOLOGY_ID = "person_water_state"
ONTOLOGY_VERSION = 1
MODEL_SCHEMA_VERSION = 1


class MobileNetV3SmallEncoder(nn.Module):
    """Shared image encoder; pretrained weights are opt-in and never downloaded implicitly."""

    output_dim = 576

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights: Any = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self.features(frames)
        return self.pool(features).flatten(1)


class DrowningTemporalModel(nn.Module):
    """Causal per-person sequence classifier with a parallel image-only head.

    Inputs are RGB tensors shaped B,T,C,H,W in the caller's shared preprocessing
    contract. The final timestep must be valid because it is the current anchor.
    Invalid/missed frames leave the recurrent hidden state unchanged.
    """

    def __init__(self, hidden_size: int = 256, pretrained_encoder: bool = False):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.encoder = MobileNetV3SmallEncoder(pretrained=pretrained_encoder)
        self.pretrained_encoder = bool(pretrained_encoder)
        self.image_head = nn.Linear(self.encoder.output_dim, len(CLASS_KEYS))
        self.temporal_cell = nn.GRUCell(self.encoder.output_dim, hidden_size)
        self.temporal_head = nn.Linear(hidden_size, len(CLASS_KEYS))
        self.hidden_size = hidden_size

    def forward_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape B,3,H,W")
        return self.image_head(self.encoder(images))

    def forward(self, frames: torch.Tensor, valid_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape B,T,3,H,W")
        batch, timesteps = frames.shape[:2]
        if timesteps == 0:
            raise ValueError("temporal input must contain at least one frame")
        if valid_mask is None:
            valid_mask = torch.ones((batch, timesteps), dtype=torch.bool, device=frames.device)
        if valid_mask.shape != (batch, timesteps):
            raise ValueError("valid_mask must have shape B,T")
        valid_mask = valid_mask.to(device=frames.device, dtype=torch.bool)
        if not torch.all(valid_mask[:, -1]):
            raise ValueError("last frame must be valid because it is the current anchor")
        encoded = self.encoder(frames.reshape(batch * timesteps, *frames.shape[2:]))
        encoded = encoded.reshape(batch, timesteps, -1)
        image_logits = self.image_head(encoded)
        hidden = encoded.new_zeros((batch, self.hidden_size))
        states = []
        for index in range(timesteps):
            candidate = self.temporal_cell(encoded[:, index], hidden)
            hidden = torch.where(valid_mask[:, index].unsqueeze(1), candidate, hidden)
            states.append(hidden)
        temporal_states = torch.stack(states, dim=1)
        return {
            "image_logits": image_logits,
            "temporal_logits": self.temporal_head(temporal_states),
            "encoded_frames": encoded,
            "valid_mask": valid_mask,
        }
