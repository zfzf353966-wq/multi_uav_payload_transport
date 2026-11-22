from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

from .curriculum_stage2_env import CurriculumStage2Env


def evaluate(model: PPO, episodes: int, seed: int) -> dict[str, float]:
    env = CurriculumStage2Env()
    returns: list[float] = []
    durations: list[float] = []
    clearances: list[float] = []
    lateral_detours: list[float] = []
    successes = collisions = timeouts = geofence_exits = 0

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        minimum_clearance = math.inf
        maximum_detour = 0.0
        terminated = truncated = False
        info: dict[str, object] = {}
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            minimum_clearance = min(
                minimum_clearance, float(info["minimum_clearance"])
            )
            maximum_detour = max(maximum_detour, float(info["route_deviation"]))
        returns.append(episode_return)
        durations.append(env.step_count * env.dt)
        clearances.append(minimum_clearance)
        lateral_detours.append(maximum_detour)
        successes += int(bool(info["success"]))
        collisions += int(bool(info["collision"]))
        geofence_exits += int(bool(info["outside"]))
        timeouts += int(truncated)

    env.close()
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "success_rate": successes / episodes,
        "collision_rate": collisions / episodes,
        "timeout_rate": timeouts / episodes,
        "geofence_exit_rate": geofence_exits / episodes,
        "mean_duration_seconds": float(np.mean(durations)),
        "mean_minimum_clearance": float(np.mean(clearances)),
        "mean_maximum_lateral_detour": float(np.mean(lateral_detours)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Curriculum Stage 2: continue pure PPO with one static obstacle"
    )
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--initial-model",
        type=Path,
        default=Path("runs/curriculum_stage1_100k_v2/best/best_model.zip"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/curriculum_stage2")
    )
    arguments = parser.parse_args()
    if not arguments.initial_model.is_file():
        raise FileNotFoundError(f"Initial model not found: {arguments.initial_model}")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    probe = CurriculumStage2Env()
    check_env(probe, warn=True)
    probe.close()
    training_env = make_vec_env(
        CurriculumStage2Env, n_envs=arguments.n_envs, seed=arguments.seed
    )
    evaluation_env = make_vec_env(
        CurriculumStage2Env, n_envs=1, seed=arguments.seed + 5_000
    )
    model = PPO.load(
        arguments.initial_model,
        env=training_env,
        device=arguments.device,
        learning_rate=arguments.learning_rate,
        ent_coef=0.0,
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
        reset_num_timesteps=True,
        progress_bar=False,
        tb_log_name="stage2",
    )
    final_model_path = arguments.output_dir / "ppo_stage2_final"
    model.save(final_model_path)
    training_env.close()
    evaluation_env.close()

    best_model_path = arguments.output_dir / "best" / "best_model.zip"
    selected_model_path = best_model_path if best_model_path.exists() else Path(
        f"{final_model_path}.zip"
    )
    selected_model = PPO.load(selected_model_path, device=arguments.device)
    metrics = {
        "stage": 2,
        "task": "one-way navigation around one randomized static obstacle",
        "potential_field_used": False,
        "initialized_from": str(arguments.initial_model),
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
