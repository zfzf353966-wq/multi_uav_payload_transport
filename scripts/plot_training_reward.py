#!/usr/bin/env python3
"""Plot the PPO training reward stored in TensorBoard event files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalar(log_path: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one scalar tag and keep the newest value for duplicate steps."""
    event_files = [log_path] if log_path.is_file() else sorted(log_path.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found under: {log_path}")

    by_step: dict[int, tuple[float, float]] = {}
    available_tags: set[str] = set()
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        available_tags.update(accumulator.Tags().get("scalars", []))
        if tag not in accumulator.Tags().get("scalars", []):
            continue
        for event in accumulator.Scalars(tag):
            previous = by_step.get(event.step)
            if previous is None or event.wall_time >= previous[0]:
                by_step[event.step] = (event.wall_time, event.value)

    if not by_step:
        choices = ", ".join(sorted(available_tags)) or "none"
        raise KeyError(f"Scalar tag {tag!r} was not found. Available scalar tags: {choices}")

    ordered = sorted(by_step.items())
    steps = np.asarray([step for step, _ in ordered], dtype=np.int64)
    values = np.asarray([data[1] for _, data in ordered], dtype=np.float64)
    return steps, values


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_log = project_root / "runs" / "rl_residual_stack4" / "tensorboard"

    parser = argparse.ArgumentParser(description="Plot PPO training episode reward from TensorBoard logs")
    parser.add_argument("--log-dir", type=Path, default=default_log)
    parser.add_argument("--tag", default="rollout/ep_rew_mean")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    log_dir = arguments.log_dir.expanduser().resolve()
    output = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else log_dir / "training_reward_curve.png"
    )
    steps, rewards = load_scalar(log_dir, arguments.tag)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.plot(
        steps,
        rewards,
        color="#1565c0",
        linewidth=2.0,
        label="Mean episode reward",
    )
    axis.set_title("PPO Training Reward")
    axis.set_xlabel("Training steps")
    axis.set_ylabel("Mean episode reward")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    print(f"Loaded {len(rewards)} points from {steps[0]} to {steps[-1]} training steps")
    print(f"Raw reward: first={rewards[0]:.2f}, last={rewards[-1]:.2f}")
    print(f"Figure: {output}")


if __name__ == "__main__":
    main()
