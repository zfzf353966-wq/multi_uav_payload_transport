from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .allocator import allocate_tasks
from .config import WorldConfig, load_world_config
from .grid_planner import AStar3D, VoxelGrid
from .models import Task, VehicleRoute
from .path_oracle import PathOracle
from .task_generator import generate_tasks
from .trajectory import ScheduledTrajectory, schedule_routes
from .visualization import save_plan_figure


def _default_root() -> Path:
    configured = os.environ.get("MUPT_ROOT")
    return Path(configured) if configured else Path.cwd()


def _serialize(
    config: WorldConfig,
    tasks: list[Task],
    routes: list[VehicleRoute],
    oracle: PathOracle,
    scheduled: list[ScheduledTrajectory],
) -> dict[str, Any]:
    schedule_by_uav = {trajectory.uav_id: trajectory for trajectory in scheduled}
    return {
        "world": config.name,
        "coordinate_frame": "ENU",
        "algorithm": {
            "assignment": "heterogeneous capacitated pickup-delivery VRP (OR-Tools)",
            "global_planner": "3D A* with obstacle inflation and line-of-sight smoothing",
            "resolution": config.planning.resolution,
            "safety_margin": config.planning.safety_margin,
            "cached_astar_pairs": oracle.cached_pair_count,
            "astar_expanded_nodes": oracle.total_expanded_nodes,
            "trajectory": "collision-checked rounded path with acceleration-limited time parameterization",
            "coordination": "prioritized space-time reservation",
            "minimum_uav_separation": config.planning.minimum_uav_separation,
        },
        "tasks": [
            {
                "id": task.id,
                "pickup": task.pickup.id,
                "pickup_position": task.pickup.position,
                "delivery": task.delivery.id,
                "delivery_position": task.delivery.position,
                "quantity": task.quantity,
            }
            for task in tasks
        ],
        "routes": [
            {
                "uav": route.uav.id,
                "capacity": route.uav.capacity,
                "peak_load": route.peak_load,
                "distance_m": round(route.distance, 3),
                "estimated_time_s": round(route.estimated_time, 3),
                "stops": [
                    {
                        "label": stop.label,
                        "kind": stop.kind,
                        "task_id": stop.task_id,
                        "position": stop.position,
                        "load_after": stop.load_after,
                    }
                    for stop in route.stops
                ],
                "path": [tuple(round(value, 3) for value in point) for point in route.path],
                "trajectory_start_delay_s": round(schedule_by_uav[route.uav.id].start_delay, 3),
                "trajectory_duration_s": round(schedule_by_uav[route.uav.id].duration, 3),
                "trajectory": [
                    {
                        "t": round(sample.time, 3),
                        "position": [round(value, 4) for value in sample.position],
                        "velocity": [round(value, 4) for value in sample.velocity],
                        "acceleration": [round(value, 4) for value in sample.acceleration],
                    }
                    for sample in schedule_by_uav[route.uav.id].samples
                ],
            }
            for route in routes
        ],
        "summary": {
            "total_distance_m": round(sum(route.distance for route in routes), 3),
            "total_flight_time_s": round(sum(route.estimated_time for route in routes), 3),
            "makespan_s": round(
                max(trajectory.start_delay + trajectory.duration for trajectory in scheduled), 3
            ),
            "all_constraints_satisfied": True,
        },
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = load_world_config(config_path)
    tasks = generate_tasks(config)
    grid = VoxelGrid(config)
    oracle = PathOracle(AStar3D(grid))
    routes = allocate_tasks(config, tasks, oracle)
    scheduled = schedule_routes(config, routes, grid)

    for route in routes:
        for first, second in zip(route.path, route.path[1:]):
            if not grid.segment_is_free(first, second):
                raise RuntimeError(f"Collision validation failed for {route.uav.id}: {first} -> {second}")

    output_dir.mkdir(parents=True, exist_ok=True)
    result = _serialize(config, tasks, routes, oracle, scheduled)
    result_path = output_dir / "plan.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    save_plan_figure(config, tasks, routes, output_dir / "plan_3d.png")

    print(f"World: {config.name}")
    print(f"Tasks: {len(tasks)} | A* pairs: {oracle.cached_pair_count}")
    for route in routes:
        sequence = " -> ".join(stop.label for stop in route.stops)
        print(
            f"{route.uav.id}: capacity={route.uav.capacity}, peak_load={route.peak_load}, "
            f"distance={route.distance:.1f} m, time={route.estimated_time:.1f} s"
        )
        print(f"  {sequence}")
    for trajectory in scheduled:
        print(
            f"  {trajectory.uav_id}: trajectory delay={trajectory.start_delay:.1f} s, "
            f"duration={trajectory.duration:.1f} s"
        )
    summary = result["summary"]
    print(
        f"PASS: constraints satisfied; total={summary['total_distance_m']:.1f} m, "
        f"makespan={summary['makespan_s']:.1f} s"
    )
    print(f"Plan: {result_path}")
    print(f"Figure: {output_dir / 'plan_3d.png'}")
    return result


def main() -> None:
    root = _default_root()
    parser = argparse.ArgumentParser(description="Run the known-map multi-UAV planning baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "logistics_city.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "runs" / "traditional_baseline",
    )
    arguments = parser.parse_args()
    run(arguments.config, arguments.output_dir)


if __name__ == "__main__":
    main()
