from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from .rl_avoidance_env import UAVLocalAvoidanceEnv


def evaluate(model: PPO | None, episodes: int, seed: int) -> dict[str, float]:
    env = UAVLocalAvoidanceEnv()
    returns = []
    successes = 0
    collisions = 0
    geofence_exits = 0
    timeouts = 0
    outbound_completions = 0
    minimum_clearances = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_clearance = float("inf")
        terminated = truncated = False
        while not (terminated or truncated):
            if model is None:
                action = np.zeros(3, dtype=np.float32)
            else:
                action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_clearance = min(episode_clearance, info["minimum_clearance"])
        returns.append(episode_return)
        successes += int(info["success"])
        collisions += int(info["collision"])
        geofence_exits += int(info["outside"])
        timeouts += int(truncated)
        outbound_completions += int(info["outbound_complete"])
        minimum_clearances.append(episode_clearance)
    env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "geofence_exit_rate": geofence_exits / episodes,
        "timeout_rate": timeouts / episodes,
        "outbound_completion_rate": outbound_completions / episodes,
        "mean_minimum_clearance": float(np.mean(minimum_clearances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lightweight local-avoidance PPO policy")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/rl_avoidance_smoke"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    probe = UAVLocalAvoidanceEnv()
    check_env(probe, warn=True)
    probe.close()
    env = make_vec_env(UAVLocalAvoidanceEnv, n_envs=arguments.n_envs, seed=arguments.seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [128, 128]},
        seed=arguments.seed,
        device=arguments.device,
        verbose=1,
        tensorboard_log=str(arguments.output_dir / "tensorboard"),
    )
    model.learn(total_timesteps=arguments.timesteps, progress_bar=False)
    model_path = arguments.output_dir / "ppo_local_avoidance"
    model.save(model_path)
    env.close()

    # Load from disk before evaluation so the smoke test also covers model
    # serialization, which will be needed by the future ROS inference node.
    loaded = PPO.load(model_path, device=arguments.device)
    evaluation_seed = arguments.seed + 10_000
    metrics = {
        "timesteps": arguments.timesteps,
        "seed": arguments.seed,
        "potential_field_baseline": evaluate(None, arguments.eval_episodes, evaluation_seed),
        "trained_policy": evaluate(loaded, arguments.eval_episodes, evaluation_seed),
    }
    metrics_path = arguments.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model: {model_path}.zip")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
