from __future__ import annotations

from dataclasses import dataclass

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .config import WorldConfig
from .models import Point3D, RouteStop, Task, VehicleRoute
from .path_oracle import PathOracle


@dataclass(frozen=True)
class _Node:
    label: str
    kind: str
    position: Point3D
    task_id: str | None
    demand: int


def allocate_tasks(
    config: WorldConfig,
    tasks: list[Task],
    paths: PathOracle,
) -> list[VehicleRoute]:
    if not tasks:
        raise ValueError("Cannot allocate an empty task list")

    nodes = [_Node("depot", "depot", config.depot.position, None, 0)]
    task_node_indices: dict[str, tuple[int, int]] = {}
    for task in tasks:
        pickup_index = len(nodes)
        nodes.append(
            _Node(
                f"{task.id}:{task.pickup.id}",
                "pickup",
                task.pickup.position,
                task.id,
                task.quantity,
            )
        )
        delivery_index = len(nodes)
        nodes.append(
            _Node(
                f"{task.id}:{task.delivery.id}",
                "delivery",
                task.delivery.position,
                task.id,
                -task.quantity,
            )
        )
        task_node_indices[task.id] = (pickup_index, delivery_index)

    paths.preload([node.position for node in nodes])
    manager = pywrapcp.RoutingIndexManager(len(nodes), len(config.uavs), 0)
    routing = pywrapcp.RoutingModel(manager)

    time_callbacks: list[int] = []
    for vehicle_id, uav in enumerate(config.uavs):
        def transit(from_index: int, to_index: int, speed: float = uav.cruise_speed) -> int:
            first = nodes[manager.IndexToNode(from_index)].position
            second = nodes[manager.IndexToNode(to_index)].position
            return int(round(paths.get(first, second).distance / speed * 1000.0))

        callback_index = routing.RegisterTransitCallback(transit)
        time_callbacks.append(callback_index)
        routing.SetArcCostEvaluatorOfVehicle(callback_index, vehicle_id)

    demand_callback = routing.RegisterUnaryTransitCallback(
        lambda index: nodes[manager.IndexToNode(index)].demand
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback,
        0,
        [uav.capacity for uav in config.uavs],
        True,
        "Capacity",
    )
    routing.AddDimensionWithVehicleTransits(
        time_callbacks,
        0,
        10**9,
        True,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    time_dimension.SetGlobalSpanCostCoefficient(config.planning.makespan_weight)

    solver = routing.solver()
    for pickup_node, delivery_node in task_node_indices.values():
        pickup_index = manager.NodeToIndex(pickup_node)
        delivery_index = manager.NodeToIndex(delivery_node)
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        solver.Add(routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index))
        solver.Add(time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index))

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.FromSeconds(config.planning.solver_time_limit_seconds)
    parameters.log_search = False
    solution = routing.SolveWithParameters(parameters)
    if solution is None:
        raise RuntimeError("OR-Tools could not find a feasible task allocation")

    routes: list[VehicleRoute] = []
    for vehicle_id, uav in enumerate(config.uavs):
        route_stops: list[RouteStop] = []
        route_points: list[Point3D] = []
        stop_path_indices: list[int] = []
        current_load = 0
        peak_load = 0
        route_distance = 0.0
        index = routing.Start(vehicle_id)

        while True:
            node_index = manager.IndexToNode(index)
            node = nodes[node_index]
            current_load += node.demand
            peak_load = max(peak_load, current_load)
            route_stops.append(
                RouteStop(
                    node_index=node_index,
                    label=node.label,
                    kind=node.kind,
                    position=node.position,
                    task_id=node.task_id,
                    load_after=current_load,
                )
            )
            if not route_points:
                route_points.append(node.position)
            stop_path_indices.append(len(route_points) - 1)
            if routing.IsEnd(index):
                break

            next_index = solution.Value(routing.NextVar(index))
            next_node = nodes[manager.IndexToNode(next_index)]
            segment = paths.get(node.position, next_node.position)
            route_points.extend(segment.points[1:])
            route_distance += segment.distance
            index = next_index

        if current_load != 0 or peak_load > uav.capacity:
            raise RuntimeError(f"Invalid capacity result for {uav.id}")
        routes.append(
            VehicleRoute(
                uav=uav,
                stops=route_stops,
                path=route_points,
                stop_path_indices=stop_path_indices,
                distance=route_distance,
                estimated_time=route_distance / uav.cruise_speed,
                peak_load=peak_load,
            )
        )

    _validate_pickup_delivery(routes, tasks)
    return routes


def _validate_pickup_delivery(routes: list[VehicleRoute], tasks: list[Task]) -> None:
    occurrences: dict[str, list[tuple[str, str, int]]] = {task.id: [] for task in tasks}
    for route in routes:
        for order, stop in enumerate(route.stops):
            if stop.task_id:
                occurrences[stop.task_id].append((route.uav.id, stop.kind, order))

    for task_id, visits in occurrences.items():
        if len(visits) != 2:
            raise RuntimeError(f"Task {task_id} was not visited exactly twice: {visits}")
        pickup, delivery = visits
        if pickup[0] != delivery[0] or pickup[1] != "pickup" or delivery[1] != "delivery":
            raise RuntimeError(f"Task {task_id} violates pickup-delivery pairing: {visits}")
        if pickup[2] >= delivery[2]:
            raise RuntimeError(f"Task {task_id} is delivered before pickup")
