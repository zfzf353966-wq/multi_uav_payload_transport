"""Deterministic input scaling shared through the saved SB3 policy class."""
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DirectPPOFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=153)

    def forward(self, observations):
        # Empty scans become zero, avoiding 144 constant ones overwhelming
        # small relative-goal signals in the first policy layer. This computes
        # no avoidance action and is serialized as part of the PPO model.
        return torch.cat((
            observations[..., :144] - 1.0,
            observations[..., 144:147] * 5.0,
            observations[..., 147:150],
            observations[..., 150:152] - .5,
            observations[..., 152:153] - .3,
        ), dim=-1)


class TemporalDirectPPOFeatures(DirectPPOFeatures):
    """Append explicit sector closing rates from existing four-frame observations."""
    def __init__(self, observation_space):
        super().__init__(observation_space)
        self._features_dim = 153 + 108

    def forward(self, observations):
        base = super().forward(observations)
        scans = observations[..., :144].reshape(*observations.shape[:-1], 4, 36)
        valid = (scans[..., :-1, :] < .999) & (scans[..., 1:, :] < .999)
        # 20 m normalization / 0.1 s interval / 4 m/s closing-rate scale = 50.
        # Visibility transitions are masked rather than treated as huge speeds.
        rates = torch.clamp((scans[..., :-1, :] - scans[..., 1:, :]) * 50.0, -1.0, 1.0)
        rates = torch.where(valid, rates, torch.zeros_like(rates))
        return torch.cat((base, rates.flatten(start_dim=-2)), dim=-1)
