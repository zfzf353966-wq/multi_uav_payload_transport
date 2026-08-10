from pathlib import Path

from logistics_planner.allocator import allocate_tasks
from logistics_planner.config import load_world_config
from logistics_planner.grid_planner import AStar3D, VoxelGrid
from logistics_planner.path_oracle import PathOracle
from logistics_planner.task_generator import generate_tasks


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "logistics_city.yaml"


def test_generated_tasks_are_reproducible() -> None:
    config = load_world_config(CONFIG)
    assert generate_tasks(config) == generate_tasks(config)
    assert len(generate_tasks(config)) == 6


def test_assignment_and_paths_satisfy_constraints() -> None:
    config = load_world_config(CONFIG)
    tasks = generate_tasks(config)
    grid = VoxelGrid(config)
    oracle = PathOracle(AStar3D(grid))
    routes = allocate_tasks(config, tasks, oracle)

    assert len(routes) == 3
    assert sum(stop.kind == "pickup" for route in routes for stop in route.stops) == len(tasks)
    assert all(route.peak_load <= route.uav.capacity for route in routes)
    assert all(
        grid.segment_is_free(first, second)
        for route in routes
        for first, second in zip(route.path, route.path[1:])
    )
