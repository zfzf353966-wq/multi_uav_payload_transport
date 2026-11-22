from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from .pure_ppo_avoidance_env import PurePPOLocalAvoidanceEnv
from .shadow_avoidance import compute_shadow_avoidance


def potential_field_action(env: PurePPOLocalAvoidanceEnv) -> np.ndarray:
    desired_body = env._world_to_body(env._nominal_velocity())
    result = compute_shadow_avoidance(
        env.scan_history[-1],
        desired_body,
        influence_distance=6.0,
        emergency_distance=2.0,
        max_correction=2.0,
        max_speed=env.max_speed,
    )
    return np.clip(
        np.asarray(result.suggested_velocity[:2], dtype=np.float32) / env.max_speed,
        -1.0,
        1.0,
    )


def evaluate(controller: str, model: PPO | None, episodes: int, seed: int) -> dict[str, float]:
    env = PurePPOLocalAvoidanceEnv()
    returns: list[float] = []
    clearances: list[float] = []
    successes = collisions = geofence_exits = timeouts = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        episode_clearance = float("inf")
        terminated = truncated = False
        while not (terminated or truncated):
            if controller == "ppo":
                if model is None:
                    raise ValueError("PPO evaluation requires a trained model")
                action, _ = model.predict(observation, deterministic=True)
            elif controller == "potential_field":
                action = potential_field_action(env)
            else:
                raise ValueError(f"Unknown controller: {controller}")
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_clearance = min(episode_clearance, float(info["minimum_clearance"]))
        returns.append(episode_return)
        clearances.append(episode_clearance)
        successes += int(info["success"])
        collisions += int(info["collision"])
        geofence_exits += int(info["outside"])
        timeouts += int(truncated)
    env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "geofence_exit_rate": geofence_exits / episodes,
        "timeout_rate": timeouts / episodes,
        "mean_minimum_clearance": float(np.mean(clearances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train direct horizontal-velocity PPO avoidance")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/pure_ppo_direct_300k"))
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    probe = PurePPOLocalAvoidanceEnv()
    check_env(probe, warn=True)
    probe.close()
    env = make_vec_env(PurePPOLocalAvoidanceEnv, n_envs=arguments.n_envs, seed=arguments.seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        policy_kwargs={"net_arch": [256, 256]},
        seed=arguments.seed,
        device=arguments.device,
        verbose=1,
        tensorboard_log=str(arguments.output_dir / "tensorboard"),
    )
    model.learn(total_timesteps=arguments.timesteps, progress_bar=False)
    model_path = arguments.output_dir / "ppo_direct_local_avoidance"
    model.save(model_path)
    env.close()

    loaded = PPO.load(model_path, device=arguments.device)
    evaluation_seed = arguments.seed + 10_000
    metrics = {
        "algorithm": "direct horizontal-velocity PPO without potential-field input",
        "timesteps": arguments.timesteps,
        "seed": arguments.seed,
        "potential_field_baseline": evaluate(
            "potential_field", None, arguments.eval_episodes, evaluation_seed
        ),
        "pure_ppo": evaluate("ppo", loaded, arguments.eval_episodes, evaluation_seed),
    }
    metrics_path = arguments.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Model: {model_path}.zip")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
