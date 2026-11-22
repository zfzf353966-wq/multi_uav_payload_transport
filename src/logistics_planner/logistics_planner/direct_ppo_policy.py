"""A Gaussian PPO actor whose deterministic mean stays inside action bounds."""
import torch
import math
from stable_baselines3.common.distributions import DiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class BoundedMeanPolicy(ActorCriticPolicy):
    def _get_action_dist_from_latent(self, latent_pi):
        # The transform is used by sampling, prediction AND log-probability
        # evaluation. Gaussian exploration can still exceed the Box and is
        # clipped by SB3; the deterministic mean itself is always bounded.
        mean = torch.tanh(self.action_net(latent_pi) / 2.0)
        if isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean, self.log_std, latent_pi)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean, self.log_std)
        raise ValueError("BoundedMeanPolicy requires continuous Gaussian exploration")


def transfer_policy_weights(source, target):
    """Preserve the old policy exactly while zero-initializing appended features."""
    old = source.state_dict()
    new = target.state_dict()
    if old.keys() != new.keys():
        raise ValueError("policy parameter names differ during warm start")
    for name, value in old.items():
        if value.shape == new[name].shape:
            new[name] = value.detach().clone()
        elif name == "log_std" and value.ndim == 1 and new[name].ndim == 2 and value.shape[0] == new[name].shape[1]:
            # Match an upper bound on variance when expanding independent
            # Gaussian noise into state-dependent, time-correlated exploration.
            new[name] = value.unsqueeze(0).expand_as(new[name]).clone() - .5 * math.log(new[name].shape[0])
        elif name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight") and (
            value.ndim == 2 and value.shape[0] == new[name].shape[0] and value.shape[1] < new[name].shape[1]
        ):
            expanded = torch.zeros_like(new[name])
            expanded[:, :value.shape[1]] = value
            new[name] = expanded
        else:
            raise ValueError(f"unsupported warm-start shape change: {name}: {value.shape} -> {new[name].shape}")
    target.load_state_dict(new)
