"""Plot a recorded PX4 PPO flight without inventing obstacle pose history."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import yaml
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--lab", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader((args.run_dir / "control_trace.csv").open()))
    summary = json.loads((args.run_dir / "summary.json").read_text())
    values = lambda key: np.array([float(r[key]) for r in rows])
    t = values("elapsed_s")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    if (args.run_dir / "spawn.json").exists():
        spawn = json.loads((args.run_dir / "spawn.json").read_text())
        model = ET.parse(args.run_dir / "obstacles.sdf").getroot().find("model")
        for joint in model.findall("joint"):
            if not joint.get("name", "").endswith("_orbit"):
                continue
            centre = [float(v) for v in joint.findtext("pose").split()[:2]]
            link = model.find(f"link[@name='{joint.findtext('child')}']")
            point = [float(v) for v in link.findtext("pose").split()[:2]]
            radius = np.linalg.norm(np.array(point) - centre)
            ax.add_patch(Circle((spawn["position"][0] + centre[0], spawn["position"][1] + centre[1]),
                                radius, fill=False, color="#ef8a00", linestyle="--", label="Verified obstacle orbit"))
    if not args.lab:
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / "config/logistics_city.yaml").read_text())
        for obstacle in config["static_obstacles"]:
            cx, cy, _ = obstacle["center"]
            sx, sy, _ = obstacle["size"]
            ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, color="gray", alpha=.2))
        for x, y, _ in config["tree_positions"]:
            ax.add_patch(Circle((x, y), 1.6, color="green", alpha=.2))
    for source, color in [("rl", "#1565c0"), ("fallback", "#c62828"), ("route", "#757575"), ("approach", "#2e7d32")]:
        mask = np.array([r["source"] == source for r in rows])
        if mask.any():
            ax.scatter(values("x_enu")[mask], values("y_enu")[mask], s=5, color=color, label=source)
    goal = summary.get("outbound_goal_enu", [-6, -23, 6])
    ax.plot([-34, goal[0]], [-23, goal[1]], "k--", alpha=.5, label="Nominal corridor")
    ax.scatter([-34, goal[0]], [-23, goal[1]], marker="x", color="black")
    ax.set(xlabel="ENU east (m)", ylabel="ENU north (m)", title="Actual PX4 trajectory and selected control source")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8)
    axes[0, 1].plot(t, values("z_enu"), label="Actual height")
    axes[0, 1].axhline(2, ls="--", color="red", alpha=.5)
    axes[0, 1].axhline(10, ls="--", color="red", alpha=.5)
    axes[0, 1].set(title="Height", xlabel="Elapsed time (s)", ylabel="ENU z (m)")
    axes[1, 0].plot(t, values("minimum_scan_m"), label="Nearest lidar return")
    axes[1, 0].plot(t, values("down_clearance_m"), label="Downward clearance", alpha=.6)
    axes[1, 0].axhline(.8, ls="--", color="red", label="No-contact gate")
    axes[1, 0].set(title="Measured clearance", xlabel="Elapsed time (s)", ylabel="Distance (m)", ylim=(0, 21))
    axes[1, 0].legend(fontsize=8)
    actual = np.sqrt(sum(values(key) ** 2 for key in ("vx_ned", "vy_ned", "vz_ned")))
    command = np.sqrt(sum(values(key) ** 2 for key in ("command_forward", "command_left", "command_up")))
    axes[1, 1].plot(t, actual, label="Actual vehicle speed")
    axes[1, 1].plot(t, command, label="Selected command speed", alpha=.6)
    axes[1, 1].set(title="Actual and commanded speed", xlabel="Elapsed time (s)", ylabel="Speed (m/s)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    fig.suptitle(f"Direct 3-D PPO local avoidance / {'lab' if args.lab else 'city'} / {'PASS' if summary['success'] else 'FAIL'} / no APF input")
    fig.tight_layout()
    output = args.run_dir / "flight_diagnostics.png"
    fig.savefig(output, dpi=170)
    print(output)


if __name__ == "__main__":
    main()
