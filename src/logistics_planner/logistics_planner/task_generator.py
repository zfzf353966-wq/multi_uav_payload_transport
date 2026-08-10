from __future__ import annotations

import random

from .config import WorldConfig
from .models import Task


def generate_tasks(config: WorldConfig) -> list[Task]:
    """Generate a deterministic task batch from configured rooftop locations."""
    settings = config.task_settings
    if settings.count <= 0:
        raise ValueError("Task count must be positive")
    if settings.package_quantity <= 0:
        raise ValueError("Package quantity must be positive")
    if settings.package_quantity > max(uav.capacity for uav in config.uavs):
        raise ValueError("No UAV can carry the configured package quantity")

    rng = random.Random(settings.seed)
    tasks: list[Task] = []
    for task_number in range(1, settings.count + 1):
        pickup = rng.choice(config.pickups)
        delivery = rng.choice(config.deliveries)
        tasks.append(
            Task(
                id=f"task_{task_number}",
                pickup=pickup,
                delivery=delivery,
                quantity=settings.package_quantity,
            )
        )
    return tasks
