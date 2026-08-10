from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import WorldConfig
from .models import Task, VehicleRoute


def save_plan_figure(
    config: WorldConfig,
    tasks: list[Task],
    routes: list[VehicleRoute],
    output_path: str | Path,
) -> None:
    figure = plt.figure(figsize=(12, 9))
    axis = figure.add_subplot(111, projection="3d")

    for obstacle in config.obstacles:
        x = obstacle.center[0] - obstacle.size[0] / 2.0
        y = obstacle.center[1] - obstacle.size[1] / 2.0
        axis.bar3d(
            x,
            y,
            0,
            obstacle.size[0],
            obstacle.size[1],
            obstacle.size[2],
            color="#6f7782",
            alpha=0.28,
            shade=True,
        )

    colors = ("#e53935", "#1e88e5", "#43a047")
    for route, color in zip(routes, colors):
        if route.path:
            points = np.asarray(route.path)
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=2.2, label=route.uav.id)
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=12)

    pickup_positions = np.asarray([location.position for location in config.pickups])
    delivery_positions = np.asarray([location.position for location in config.deliveries])
    axis.scatter(*pickup_positions.T, marker="s", s=90, color="#fbc02d", edgecolor="black", label="pickup")
    axis.scatter(*delivery_positions.T, marker="o", s=90, color="#00acc1", edgecolor="black", label="delivery")
    axis.scatter(*config.depot.position, marker="^", s=140, color="black", label="depot")

    axis.set_xlim(config.minimum[0], config.maximum[0])
    axis.set_ylim(config.minimum[1], config.maximum[1])
    axis.set_zlim(0, config.maximum[2])
    axis.set_xlabel("East x [m]")
    axis.set_ylabel("North y [m]")
    axis.set_zlabel("Altitude z [m]")
    axis.set_title(f"Known-map task allocation and 3D A* ({len(tasks)} tasks)")
    axis.legend(loc="upper left")
    axis.view_init(elev=32, azim=-58)
    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
