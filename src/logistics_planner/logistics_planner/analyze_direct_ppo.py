"""Reproduce direct PPO failures and export geometric/control diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from .direct_ppo_contract import action_to_velocity, project_safe_velocity
from .direct_ppo_env import DirectPPOEnv


def rollout(model, stage, seed, speed_scale=1.0, *, reward_version=2, safety_guards=False):
    from .single_uav_avoidance_test import direct_velocity_decision
    mapping = getattr(model, "velocity_mapping", "radial")
    env = DirectPPOEnv(stage, reward_version=reward_version, safety_guards=safety_guards,
                       velocity_mapping=mapping)
    observation, _ = env.reset(seed=seed)
    route = env.goal - env.start
    length = float(np.linalg.norm(route[:2]))
    along = np.array([route[0] / length, route[1] / length, 0.0])
    lateral = np.array([-along[1], along[0], 0.0])
    obstacle = env.obstacles[0]
    initial = dict(seed=seed, heading_rad=env.heading,
                   route_bearing_body_deg=math.degrees(math.atan2(route[1], route[0]) - env.heading) % 360,
                   route_length_m=length, goal_height_m=float(env.goal[2]),
                   obstacle_radius_m=obstacle.radius,
                   orbit_radius_m=env.orbits[0][1], orbit_omega=env.orbits[0][3],
                   obstacle_along_m=float((obstacle.position - env.start) @ along))
    rows, total_return = [], 0.0
    minimum_clearance = math.inf
    while True:
        action, _ = model.predict(observation, deterministic=True)
        action = action * speed_scale
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(observation)
            raw_action = model.policy.get_distribution(tensor).distribution.mean.cpu().numpy()[0]
        before = env.position.copy()
        old_distance = env._goal_distance()
        old_action = env.previous_action.copy()
        scan = env.scan_history[-1].copy()
        obstacle_relative = obstacle.position - before
        passed_before = 0 in env.passed_obstacles
        obstacle_before = obstacle.position.copy()
        relative_body = env._world_to_body(obstacle_relative)
        policy_body = action_to_velocity(action, mapping)
        actual_body, guards = project_safe_velocity(policy_body, float(before[2]), env._down_clearance()) if safety_guards else (policy_body, ())
        command = env._body_to_world(actual_body)
        runtime_decision = direct_velocity_decision(policy_body, 0.0, scan, 0.0)
        closing = float(command @ obstacle_relative) / max(float(np.linalg.norm(obstacle_relative)), 1e-6)
        obs_clearance = env._clearance()
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += reward
        minimum_clearance = min(minimum_clearance, info["minimum_clearance"])
        direction = env._world_to_body(env.goal - env.position)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        progress_reward = 12 * (old_distance - info["goal_distance"])
        alignment_reward = .6 * float(action @ direction)
        clearance_penalty = .6 * (1.5 - max(info["minimum_clearance"], 0.0)) ** 2 if info["minimum_clearance"] < 1.5 else 0.0
        ttc_penalty = .3 * (2 - info["minimum_ttc"]) ** 2 if info["minimum_ttc"] < 2 else 0.0
        offset = env.position - env.start
        obstacle_offset = obstacle.position - env.start
        rows.append(dict(
            t=env.step_count * env.dt, x=float(env.position[0]), y=float(env.position[1]), z=float(env.position[2]),
            along=float(offset @ along), lateral=float(offset @ lateral),
            obstacle_along=float(obstacle_offset @ along), obstacle_lateral=float(obstacle_offset @ lateral),
            obstacle_z=float(obstacle.position[2]), obstacle_radius=obstacle.radius,
            obstacle_before_x=float(obstacle_before[0]), obstacle_before_y=float(obstacle_before[1]),
            obstacle_vx=float(obstacle.velocity[0]), obstacle_vy=float(obstacle.velocity[1]),
            reward_mode=info["reward_mode"], passed_before=int(passed_before),
            safety_guarded=int(bool(guards)), policy_command_speed=float(np.linalg.norm(policy_body)),
            clearance=info["minimum_clearance"], clearance_before=obs_clearance,
            goal_distance=info["goal_distance"], speed=float(np.linalg.norm(env.velocity)),
            command_speed=float(np.linalg.norm(command)), command_along=float(command @ along),
            runtime_brake_would_trigger=int(runtime_decision.source == "fallback"),
            command_lateral=float(command @ lateral), command_up=float(command[2]),
            velocity_up=float(env.velocity[2]), command_closing=closing,
            action_forward=float(action[0]), action_left=float(action[1]), action_up=float(action[2]),
            raw_forward=float(raw_action[0]), raw_left=float(raw_action[1]), raw_up=float(raw_action[2]),
            min_scan_before=float(scan.min()), visible_sectors_before=int(np.sum(scan < 19.99)),
            obstacle_elevation_before_deg=math.degrees(math.atan2(float(relative_body[2]), float(np.linalg.norm(relative_body[:2])))),
            down_clearance=env._down_clearance(), ttc=info["minimum_ttc"],
            progress_reward=progress_reward, alignment_reward=alignment_reward,
            clearance_penalty=clearance_penalty, ttc_penalty=ttc_penalty,
            height_penalty=.08 * abs(float(env.goal[2] - env.position[2])),
            action_change=float(np.linalg.norm(action - old_action)), reward=reward,
        ))
        if terminated or truncated:
            break
    near = [r for r in rows if r["clearance_before"] < 2.0]
    approach = [r for r in near if r["along"] < r["obstacle_along"]]
    initial.update(
        outcome="collision" if info["collision"] else "outside" if info["outside"] else "success" if info["success"] else "timeout",
        duration_s=env.step_count * env.dt, final_goal_distance=info["goal_distance"],
        minimum_clearance=minimum_clearance, episode_return=total_return,
        maximum_lateral_m=max(abs(r["lateral"]) for r in rows),
        maximum_vertical_m=max(abs(r["z"] - 6.0) for r in rows),
        blind_ticks_near=len([r for r in near if r["visible_sectors_before"] == 0]),
        near_ticks=len(near),
        mean_approach_command_speed=float(np.mean([r["command_speed"] for r in approach])) if approach else None,
        near_closing_fraction=float(np.mean([r["command_closing"] > .1 for r in near])) if near else None,
        near_horizontal_saturation_fraction=float(np.mean([abs(r["action_forward"]) >= .999 and abs(r["action_left"]) >= .999 for r in near])) if near else None,
        near_braking_fraction=float(np.mean([r["command_speed"] < 1.5 for r in near])) if near else None,
        runtime_brake_fraction=float(np.mean([r["runtime_brake_would_trigger"] for r in rows])),
        near_positive_nonterminal_reward_fraction=float(np.mean([r["reward"] > 0 for r in near[:-1]])) if len(near) > 1 else None,
        final=rows[-1],
    )
    env.close()
    return initial, rows


def plots(episodes, traces, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    failures = [e for e in episodes if e["outcome"] != "success"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for episode in episodes:
        rows = traces[episode["seed"]]
        failed = episode["outcome"] != "success"
        color = "#c62828" if episode["outcome"] == "collision" else "#ef8a00" if failed else "#1565c0"
        delta = episode["obstacle_along_m"]
        axes[0].plot([r["along"] - delta for r in rows], [r["lateral"] for r in rows], color=color, alpha=.75 if failed else .05)
        axes[1].plot([r["along"] - delta for r in rows], [r["z"] - r["obstacle_z"] for r in rows], color=color, alpha=.75 if failed else .05)
    for axis in axes:
        axis.add_patch(Circle((0, 0), 1.15 + .45, color="gray", alpha=.2))
        axis.axvline(0, color="gray", ls="--", lw=.8)
        axis.set_xlabel("Along-track distance from obstacle centre (m)")
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Lateral offset (m)")
    axes[1].set_ylabel("Height relative to obstacle (m)")
    fig.suptitle("PPO trajectories: blue=success, red=collision, orange=timeout; shaded radius=largest inflated sphere")
    fig.tight_layout()
    fig.savefig(output / "trajectories.png", dpi=160)
    plt.close(fig)

    examples = [next((e for e in failures if e["outcome"] == "collision"), None),
                next((e for e in failures if e["outcome"] == "timeout"), None)]
    examples = [e for e in examples if e]
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 3, figsize=(15, 4 * len(examples)), squeeze=False)
    for axis_row, episode in zip(axes, examples):
        rows = traces[episode["seed"]]
        t = [r["t"] for r in rows]
        axis_row[0].plot(t, [r["clearance"] for r in rows], label="Airframe clearance (m)")
        axis_row[0].plot(t, [r["min_scan_before"] for r in rows], label="Min observed scan (m)")
        axis_row[0].axhline(0, color="red", ls="--")
        axis_row[1].plot(t, [r["command_along"] for r in rows], label="Along command (m/s)")
        axis_row[1].plot(t, [r["command_lateral"] for r in rows], label="Lateral command (m/s)")
        axis_row[1].plot(t, [r["command_up"] for r in rows], label="Up command (m/s)")
        axis_row[2].plot(t, [r["progress_reward"] + r["alignment_reward"] for r in rows], label="V2 progress + direction reference")
        axis_row[2].plot(t, [r["clearance_penalty"] + r["ttc_penalty"] for r in rows], label="V2 clearance + TTC reference")
        for axis in axis_row:
            axis.set_xlabel("Time (s)")
            axis.set_title(f"Seed {episode['seed']} / {episode['outcome']}")
            axis.legend(fontsize=8)
            axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "failure_details.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", default="static")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-version", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--safety-guards", action="store_true")
    parser.add_argument("--speed-scale", type=float, default=1.0,
                        help="Diagnostic counterfactual: scale all policy velocity commands")
    args = parser.parse_args()
    if not 0 < args.speed_scale <= 1:
        parser.error("speed-scale must be in (0, 1]")
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes, traces = [], {}
    for seed in range(args.seed, args.seed + args.episodes):
        episode, rows = rollout(model, args.stage, seed, args.speed_scale,
                                reward_version=args.reward_version, safety_guards=args.safety_guards)
        episodes.append(episode)
        traces[seed] = rows
        if episode["outcome"] != "success":
            with (args.output_dir / f"trace_{seed}.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    counts = {key: sum(e["outcome"] == key for e in episodes) for key in ("success", "collision", "timeout", "outside")}
    result = dict(model=str(args.model), stage=args.stage, seed=args.seed, episodes=args.episodes,
                  diagnostic_speed_scale=args.speed_scale,
                  reward_version=args.reward_version, safety_guards=args.safety_guards,
                  reward_components_reference_version=2,
                  counts=counts, results=episodes)
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: finite_json(item) for key, item in value.items()}
        if isinstance(value, list):
            return [finite_json(item) for item in value]
        return value
    (args.output_dir / "analysis.json").write_text(json.dumps(finite_json(result), indent=2, allow_nan=False) + "\n")
    plots(episodes, traces, args.output_dir)
    print(json.dumps(counts), flush=True)
    print(f"Diagnostics: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
