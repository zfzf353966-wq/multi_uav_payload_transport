"""A* legs preserve every pickup/delivery service stop after local detours."""
import json
from pathlib import Path
from .config import load_world_config
from .grid_planner import AStar3D, VoxelGrid
from .trajectory import staging_position


def validate_delivery_plan(plan):
    """起飞前检查同机先取后送、容量、目的地归属及空载返航。"""
    tasks={t['id']:t for t in plan['tasks']}
    picked=set();delivered=set()
    owners={}
    for route in plan['routes']:
        onboard=set()
        if not route['delivery_legs'] or route['delivery_legs'][-1]['kind']!='return':
            raise ValueError('Every route must return home')
        for leg in route['delivery_legs']:
            task=leg['task_id']
            if leg['kind']=='return':
                if onboard:raise ValueError('Returning with undelivered cargo')
                continue
            if task not in tasks:raise ValueError('Unknown task')
            if leg['kind']=='pickup':
                if task in picked:raise ValueError('Task picked up twice')
                picked.add(task);onboard.add(task)
                if sum(tasks[t]['quantity'] for t in onboard)>route['capacity']:
                    raise ValueError('Payload capacity exceeded')
            elif leg['kind']=='delivery':
                if task not in onboard:raise ValueError('Delivery has no matching pickup on this UAV')
                onboard.remove(task);delivered.add(task)
                if plan.get('algorithm',{}).get('exclusive_delivery_uav'):
                    destination=tasks[task]['delivery']
                    previous=owners.setdefault(destination,route['uav'])
                    if previous!=route['uav']:raise ValueError('Delivery point assigned to multiple UAVs')
            else:
                raise ValueError('Unknown service kind')
    if picked!=set(tasks) or delivered!=set(tasks):
        raise ValueError('Every task must be picked up and delivered exactly once')


def add_delivery_legs(plan, config_path):
    config = load_world_config(config_path)
    planner = AStar3D(VoxelGrid(config))
    for route in plan['routes']:
        index = next(i for i, u in enumerate(config.uavs) if u.id == route['uav'])
        staging = staging_position(config.uav_spawn_points[index], index)
        current = staging
        legs = []
        for stop in route['stops']:
            if stop['kind'] == 'depot':
                continue
            target = tuple(stop['position'])
            path = planner.plan(current, target).points
            legs.append(dict(kind=stop['kind'], task_id=stop['task_id'], target=target, path=path))
            current = target
        # Empty return: climb first, cruise above rooftops, descend only over
        # the aircraft's own pad. Preserve these A* section boundaries.
        altitude=max(current[2],26.+2.*index)
        high_departure=(current[0],current[1],altitude)
        high_home=(staging[0],staging[1],altitude)
        path=[current]
        for target in (high_departure,high_home,staging):
            path.extend(planner.plan(path[-1],target).points[1:])
        legs.append(dict(kind='return', task_id=None, target=staging,
                         path=path,return_cruise_altitude_m=altitude))
        route['delivery_legs'] = legs
    plan['world'] = 'logistics_delivery_ppo'
    plan['algorithm']['local_planner'] = 'PPO direct FLU 3D velocity with braking and altitude guards'
    plan['algorithm']['execution'] = 'actual-position A* waypoint progress; live fleet separation protection'
    plan['algorithm']['coordination'] = 'live predictive separation holds and service-point ownership; offline timing guarantees not assumed after PPO detours'
    plan['algorithm']['trajectory'] = 'A* delivery legs; original timed samples retained only for baseline compatibility'
    validate_delivery_plan(plan)
    return plan


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = add_delivery_legs(json.loads(args.plan.read_text()), args.config)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
