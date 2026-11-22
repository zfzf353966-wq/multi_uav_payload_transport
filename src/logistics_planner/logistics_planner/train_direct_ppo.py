"""Train/evaluate direct 3-D PPO; curriculum gates use a disjoint holdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from .direct_ppo_env import DirectPPOEnv, STAGES
from .direct_ppo_features import DirectPPOFeatures, TemporalDirectPPOFeatures
from .direct_ppo_policy import BoundedMeanPolicy, transfer_policy_weights
from .direct_ppo_contract import action_to_velocity


def evaluate(model, stage, episodes, seed, *, randomize=False, round_trip=False, reward_version=2, safety_guards=False, history_warmup=False, velocity_mapping="radial", direction_brake=False, lidar_profile="legacy"):
    env = DirectPPOEnv(stage, randomize=randomize, round_trip=round_trip,
                       reward_version=reward_version, safety_guards=safety_guards,
                       history_warmup=history_warmup, velocity_mapping=velocity_mapping, direction_brake=direction_brake,
                       lidar_profile=lidar_profile)
    counts = dict(success=0, collision=0, outside=0, timeout=0)
    returns, clearances, durations = [], [], []
    saturation_ticks = total_ticks = near_ticks = near_saturated = safe_successes = 0
    guarded_ticks = 0
    direction_brake_ticks = 0
    near_speeds = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        total, clearance = 0.0, float("inf")
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_ticks += 1
            guarded_ticks += int(info["safety_guarded"])
            direction_brake_ticks += int(info["direction_braked"])
            saturated = bool(np.any(np.abs(action) >= .999))
            saturation_ticks += int(saturated)
            if info["minimum_clearance"] < 2.0:
                near_ticks += 1
                near_saturated += int(saturated)
                near_speeds.append(float(np.linalg.norm(action_to_velocity(action, velocity_mapping))))
            total += reward
            clearance = min(clearance, info["minimum_clearance"])
            if terminated or truncated:
                for key in ("success", "collision", "outside"):
                    counts[key] += int(info[key])
                counts["timeout"] += int(truncated)
                break
        returns.append(total)
        clearances.append(clearance)
        durations.append(env.step_count * env.dt)
        safe_successes += int(info["success"] and clearance >= .35)
    env.close()
    finite = [v for v in clearances if np.isfinite(v)]
    return {"episodes": episodes, "seed_start": seed,
            **{f"{key}_rate": value / episodes for key, value in counts.items()},
            "mean_return": float(np.mean(returns)),
            "mean_duration_s": float(np.mean(durations)),
            "mean_minimum_clearance_m": float(np.mean(finite)) if finite else None,
               "success_with_clearance_rate": safe_successes / episodes,
               "clearance_below_035_rate": sum(v < .35 for v in clearances) / episodes,
               "action_saturation_fraction": saturation_ticks / max(1, total_ticks),
               "near_action_saturation_fraction": near_saturated / max(1, near_ticks),
               "near_command_speed_mean": float(np.mean(near_speeds)) if near_speeds else None,
               "safety_guard_fraction": guarded_ticks / max(1, total_ticks),
               "direction_brake_fraction": direction_brake_ticks / max(1, total_ticks)}


def checkpoint_score(result):
    feasible = (result["success_rate"] >= .95 and result["collision_rate"] <= .01
                and result["outside_rate"] == 0)
    return (feasible, result["success_rate"], -result["collision_rate"],
            -result["outside_rate"], result["mean_return"])


class SelectCheckpoint(BaseCallback):
    def __init__(self, stage, output, interval, randomize, round_trip,
                 reward_version=2, validation_episodes=30, validation_seed=50000, safety_guards=False,
                 history_warmup=False, velocity_mapping="radial", direction_brake=False, lidar_profile="legacy"):
        super().__init__()
        self.stage, self.output, self.interval = stage, output, interval
        self.randomize, self.round_trip = randomize, round_trip
        self.last_check = 0
        self.best_score = None
        self.records = []
        self.reward_version = reward_version
        self.validation_episodes = validation_episodes
        self.validation_seed = validation_seed
        self.best_steps = 0
        self.safety_guards = safety_guards
        self.history_warmup = history_warmup
        self.velocity_mapping = velocity_mapping
        self.direction_brake = direction_brake
        self.lidar_profile = lidar_profile

    def _on_training_start(self):
        self._check()

    def _on_step(self):
        if self.num_timesteps - self.last_check < self.interval:
            return True
        self.last_check = self.num_timesteps
        self._check()
        return True

    def _check(self):
        result = evaluate(self.model, self.stage, self.validation_episodes, self.validation_seed,
                          randomize=self.randomize, round_trip=self.round_trip,
                          reward_version=self.reward_version, safety_guards=self.safety_guards,
                          history_warmup=self.history_warmup, velocity_mapping=self.velocity_mapping,
                          direction_brake=self.direction_brake, lidar_profile=self.lidar_profile)
        score = checkpoint_score(result)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_steps = self.num_timesteps
            self.model.save(self.output / "best_model")
        self.records.append({"timesteps": self.num_timesteps, **result})
        (self.output / "validation.json").write_text(json.dumps(self.records, indent=2) + "\n")
        print(f"VALIDATION {self.stage} steps={self.num_timesteps}: {result}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="navigation")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=100000)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--validation-seed", type=int, default=50000)
    parser.add_argument("--reward-version", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--policy", choices=("gaussian", "bounded"), default=None)
    parser.add_argument("--features", choices=("auto", "basic", "temporal"), default="auto")
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--use-sde", action="store_true")
    parser.add_argument("--sde-sample-freq", type=int, default=8)
    parser.add_argument("--history-warmup", action="store_true")
    parser.add_argument("--velocity-mapping", choices=("radial", "vertical_priority"))
    parser.add_argument("--hard-case-probability", type=float, default=0.0,
                        help="Training-only near-encounter oversampling; validation/holdout stay unchanged")
    parser.add_argument("--direction-brake", action="store_true")
    parser.add_argument("--lidar-profile", choices=("legacy", "gazebo"))
    parser.add_argument("--obstacle-presence-probability", type=float, default=1.0)
    parser.add_argument("--safety-guards", action="store_true",
                        help="Apply the same altitude/downward projection used in live direct control")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--round-trip", action="store_true")
    parser.add_argument("--evaluate-only", type=Path)
    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 1e-4 if args.initial_model else 3e-4
    if min(args.timesteps, args.eval_episodes, args.n_envs, args.validation_episodes) <= 0:
        parser.error("timesteps, eval-episodes and n-envs must be positive")
    torch.set_num_threads(1)
    if args.velocity_mapping is None or args.lidar_profile is None:
        source_path = args.evaluate_only or args.initial_model
        source_model = PPO.load(source_path, device="cpu") if source_path else None
        if args.velocity_mapping is None:
            args.velocity_mapping = getattr(source_model, "velocity_mapping", "radial")
        if args.lidar_profile is None:
            args.lidar_profile = getattr(source_model, "lidar_profile", "legacy")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = dict(stage=args.stage, randomize=args.randomize, round_trip=args.round_trip,
                      reward_version=args.reward_version, safety_guards=args.safety_guards,
                      history_warmup=args.history_warmup, velocity_mapping=args.velocity_mapping,
                      hard_case_probability=args.hard_case_probability, direction_brake=args.direction_brake,
                      lidar_profile=args.lidar_profile, obstacle_presence_probability=args.obstacle_presence_probability)
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    probe = DirectPPOEnv(**env_kwargs)
    check_env(probe)
    probe.close()
    if args.evaluate_only:
        model = PPO.load(args.evaluate_only, device="cpu")
        selected = args.evaluate_only
    else:
        env = make_vec_env(DirectPPOEnv, n_envs=args.n_envs, seed=args.seed,
                           env_kwargs=env_kwargs, monitor_dir=str(args.output_dir / "monitor"))
        if args.initial_model:
            custom = ({"policy_class": BoundedMeanPolicy if args.policy == "bounded" else ActorCriticPolicy}
                      if args.policy else None)
            model = PPO.load(args.initial_model, env=env, device="cpu",
                             custom_objects=custom,
                             seed=args.seed,
                             learning_rate=args.learning_rate, ent_coef=.005,
                             tensorboard_log=str(args.output_dir / "tensorboard"))
            if args.reset_optimizer or args.features != "auto" or (args.use_sde and not model.use_sde):
                old_model = model
                kwargs = dict(old_model.policy_kwargs)
                if args.features != "auto":
                    kwargs["features_extractor_class"] = (TemporalDirectPPOFeatures if args.features == "temporal" else DirectPPOFeatures)
                model = PPO(old_model.policy_class, env, learning_rate=args.learning_rate,
                            n_steps=old_model.n_steps, batch_size=old_model.batch_size,
                            n_epochs=old_model.n_epochs, gamma=old_model.gamma,
                            gae_lambda=old_model.gae_lambda, clip_range=old_model.clip_range,
                            ent_coef=old_model.ent_coef, vf_coef=old_model.vf_coef,
                            max_grad_norm=old_model.max_grad_norm,
                            use_sde=args.use_sde or old_model.use_sde,
                            sde_sample_freq=args.sde_sample_freq if args.use_sde else old_model.sde_sample_freq,
                            policy_kwargs=kwargs, seed=args.seed, device="cpu", verbose=0,
                            tensorboard_log=str(args.output_dir / "tensorboard"))
                transfer_policy_weights(old_model.policy, model.policy)
        else:
            policy = BoundedMeanPolicy if args.policy == "bounded" else ActorCriticPolicy
            model = PPO(policy, env, learning_rate=args.learning_rate, n_steps=256,
                        use_sde=args.use_sde, sde_sample_freq=args.sde_sample_freq,
                        batch_size=256, n_epochs=8, gamma=.99, gae_lambda=.95,
                        ent_coef=.005, policy_kwargs={"net_arch": [128, 128],
                                                     "features_extractor_class": TemporalDirectPPOFeatures if args.features == "temporal" else DirectPPOFeatures},
                        device="cpu", seed=args.seed, verbose=0,
                        tensorboard_log=str(args.output_dir / "tensorboard"))
        model.velocity_mapping = args.velocity_mapping
        model.lidar_profile = args.lidar_profile
        callback = SelectCheckpoint(args.stage, args.output_dir, 25000,
                                    args.randomize, args.round_trip, args.reward_version,
                                    args.validation_episodes, args.validation_seed, args.safety_guards,
                                    args.history_warmup, args.velocity_mapping, args.direction_brake, args.lidar_profile)
        model.learn(total_timesteps=args.timesteps, callback=callback, reset_num_timesteps=True)
        model.save(args.output_dir / "final_model")
        selected = args.output_dir / "best_model.zip"
        if not selected.exists():
            selected = args.output_dir / "final_model.zip"
        env.close()
        model = PPO.load(selected, device="cpu")
    if model.action_space.shape != (3,) or model.observation_space.shape != (153,):
        raise ValueError("model does not implement the direct 3-D PPO contract (153 observations, 3 actions)")
    result = evaluate(model, args.stage, args.eval_episodes, args.eval_seed,
                      randomize=args.randomize, round_trip=args.round_trip,
                      reward_version=args.reward_version, safety_guards=args.safety_guards,
                      history_warmup=args.history_warmup, velocity_mapping=args.velocity_mapping,
                      direction_brake=args.direction_brake, lidar_profile=args.lidar_profile)
    gate = result["success_rate"] >= .95 and result["collision_rate"] <= .01 and result["outside_rate"] == 0
    metrics = dict(algorithm="direct_3d_velocity_ppo", potential_field_used=False,
                   reward_version=args.reward_version,
                   safety_guards=args.safety_guards,
                   history_warmup=args.history_warmup,
                   velocity_mapping=args.velocity_mapping,
                   model_velocity_mapping=getattr(model, "velocity_mapping", "radial"),
                   training_hard_case_probability=args.hard_case_probability,
                   evaluation_hard_case_probability=0.0,
                   direction_brake=args.direction_brake,
                   lidar_profile=args.lidar_profile,
                   training_obstacle_presence_probability=args.obstacle_presence_probability,
                   evaluation_obstacle_presence_probability=1.0,
                   evaluation_mode=("direction_brake_and_altitude_downward_guarded" if args.direction_brake and args.safety_guards
                                    else "direction_brake_guarded" if args.direction_brake
                                    else "altitude_downward_guarded" if args.safety_guards else "raw_policy"),
                   learning_rate=None if args.evaluate_only else args.learning_rate,
                   policy_class=type(model.policy).__name__,
                   validation_episodes=args.validation_episodes, validation_seed=args.validation_seed,
                   selected_training_steps=None if args.evaluate_only else callback.best_steps,
                   features_extractor=type(model.policy.features_extractor).__name__,
                   optimizer_restarted=bool(args.initial_model and (args.reset_optimizer or args.features != "auto" or args.use_sde)),
                   use_sde=model.use_sde, sde_sample_freq=model.sde_sample_freq,
                   stage=args.stage, observation_size=153, action_frame="FLU",
                   action_scale_mps=[2.0, 2.0, 1.0], maximum_speed_mps=2.0,
                   randomize=args.randomize, round_trip=args.round_trip,
                   timesteps=0 if args.evaluate_only else args.timesteps,
                   seed=args.seed, initialized_from=str(args.initial_model) if args.initial_model else None,
                   selected_model=str(selected), evaluation=result, gate_passed=gate)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
