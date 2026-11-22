from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from .curriculum_stage1_env import CurriculumStage1Env


def evaluate(model: PPO, episodes: int, seed: int) -> dict[str, float]:
    env = CurriculumStage1Env()
    returns: list[float] = []
    durations: list[float] = []
    final_distances: list[float] = []
    successes = timeouts = geofence_exits = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        terminated = truncated = False
        info: dict[str, object] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        returns.append(episode_return)
        durations.append(env.step_count * env.dt)
        final_distances.append(float(np.linalg.norm(env.goal - env.position)))
        successes += int(bool(info["success"]))
        geofence_exits += int(bool(info["outside"]))
        timeouts += int(truncated)

    env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "timeout_rate": timeouts / episodes,
        "geofence_exit_rate": geofence_exits / episodes,
        "mean_duration_seconds": float(np.mean(durations)),
        "mean_final_goal_distance": float(np.mean(final_distances)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum Stage 1: train pure PPO for unobstructed goal navigation"
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/curriculum_stage1")
    )
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    probe = CurriculumStage1Env()
    check_env(probe, warn=True)
    probe.close()

    training_env = make_vec_env(
        CurriculumStage1Env, n_envs=arguments.n_envs, seed=arguments.seed
    )
    evaluation_env = make_vec_env(
        CurriculumStage1Env, n_envs=1, seed=arguments.seed + 5_000
    )
    model = PPO(
        "MlpPolicy",
        training_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.002,
        policy_kwargs={"net_arch": [128, 128]},
        seed=arguments.seed,
        device=arguments.device,
        verbose=1,
        tensorboard_log=str(arguments.output_dir / "tensorboard"),
    )
    callback = EvalCallback(
        evaluation_env,
        best_model_save_path=str(arguments.output_dir / "best"),
        log_path=str(arguments.output_dir / "evaluation"),
        eval_freq=max(1, 10_000 // arguments.n_envs),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )
    model.learn(
        total_timesteps=arguments.timesteps,
        callback=callback,
        progress_bar=False,
        tb_log_name="stage1",
    )
    final_model_path = arguments.output_dir / "ppo_stage1_final"
    model.save(final_model_path)
    training_env.close()
    evaluation_env.close()

    best_model_path = arguments.output_dir / "best" / "best_model.zip"
    selected_model_path = best_model_path if best_model_path.exists() else Path(
        f"{final_model_path}.zip"
    )
    selected_model = PPO.load(selected_model_path, device=arguments.device)
    metrics = {
        "stage": 1,
        "task": "unobstructed one-way navigation with direct PPO velocity control",
        "potential_field_used": False,
        "timesteps": arguments.timesteps,
        "seed": arguments.seed,
        "selected_model": str(selected_model_path),
        "evaluation": evaluate(
            selected_model, arguments.eval_episodes, arguments.seed + 10_000
        ),
    }
    metrics_path = arguments.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Selected model: {selected_model_path}")
    print(f"Metrics: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
