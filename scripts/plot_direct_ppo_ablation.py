"""Plot validation curves and matched-scenario direct PPO comparisons."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    arms = {"control": "Continue unchanged", "reward": "Reward only", "bounded": "Bounded mean only"}
    metrics = [("success_rate", "Success (%)"), ("collision_rate", "Collision (%)"),
               ("action_saturation_fraction", "Action saturation (%)"),
               ("success_with_clearance_rate", "Success with >=0.35 m body clearance (%)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for arm, label in arms.items():
        records = json.loads((args.root / arm / "validation.json").read_text())
        for axis, (key, title) in zip(axes.flat, metrics):
            axis.plot([r["timesteps"] for r in records], [100 * r[key] for r in records], marker=".", label=label)
            axis.set_title(title)
            axis.set_xlabel("Additional training steps")
            axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Direct 3-D PPO: same initialization / 100 fixed validation scenarios / one training seed")
    fig.tight_layout()
    fig.savefig(args.root / "validation_curves.png", dpi=160)
    plt.close(fig)

    final_arms = {"frozen_baseline": "Frozen", **arms}
    results = [json.loads((args.root / arm / "metrics.json").read_text())["evaluation"] for arm in final_arms]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [100 * r[key] for r in results]
        axis.bar(range(len(values)), values, color=["gray", "#1565c0", "#ef8a00", "#2e7d32"])
        axis.set_xticks(range(len(values)), ["Frozen", "Control", "Reward", "Bounded"])
        axis.set_title(title)
        axis.set_ylim(0, 110 if key != "collision_rate" else max(values) * 1.3 + .5)
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.1f}", ha="center", va="bottom")
        axis.grid(axis="y", alpha=.25)
    fig.suptitle("Matched 200-episode holdout (seeds 700000-700199); exploratory single-seed comparison")
    fig.tight_layout()
    fig.savefig(args.root / "holdout_comparison.png", dpi=160)
    plt.close(fig)
    print(args.root / "validation_curves.png")
    print(args.root / "holdout_comparison.png")


if __name__ == "__main__":
    main()
