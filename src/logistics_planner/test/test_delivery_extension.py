import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from logistics_planner.city_dynamic_obstacles import generate_specs, orbit_is_safe, orbit_position,write_world
from logistics_planner.direct_ppo_contract import build_observation, project_safe_velocity
from logistics_planner.delivery_ppo_env import DeliveryPPOEnv
from logistics_planner.delivery_plan import validate_delivery_plan
from logistics_planner.delivery_control import novel_obstacle_minimum,limit_tracking_speed,project_requested_components,dynamic_radial_bounds,physical_segment_free,relevant_dynamic_minimum
from logistics_planner.city_ppo_env import CityPPOEnv
from logistics_planner.grid_planner import VoxelGrid
from logistics_planner.config import load_world_config
from logistics_planner.direct_ppo_contract import project_directional_velocity
from logistics_planner.delivery_audit import audit_geometry

ROOT=Path(__file__).resolve().parents[3]


def fixture_plan():
    return dict(routes=[dict(delivery_legs=[dict(kind='pickup',task_id='task_1',target=[-21,-11,14],
        path=[[-34,-23,3],[-34,-18,14],[-21,-11,14]])]),
        dict(delivery_legs=[dict(kind='pickup',task_id='task_2',target=[0,-12,20],
        path=[[-30,-23,10.5],[-30,-18,20],[0,-12,20]])])])


def test_random_3d_orbits_cover_city_and_preserve_swept_clearance():
    config=yaml.safe_load((ROOT/'config/logistics_city.yaml').read_text())
    for seed in range(8):
        specs=generate_specs(config,fixture_plan(),seed=seed)
        assert len(specs)==10 and sum(s['category']=='departure' for s in specs)==2
        random=[s for s in specs if s['category']=='city_random']
        assert {(s['center'][0]>0,s['center'][1]>0) for s in random}=={(0,0),(1,0),(0,1),(1,1)}
        for s in specs:
            assert orbit_is_safe(s,config)
            assert .35<=s['amplitude'][2]<=1.
            positions=np.array([orbit_position(s,t) for t in np.linspace(0,50,100)])
            assert np.ptp(positions[:,2])>.5
        for s in specs[:2]:
            assert min(np.linalg.norm(np.array(s['center'])-r['delivery_legs'][0]['path'][0])
                       for r in fixture_plan()['routes'])<=12
    assert generate_specs(config,fixture_plan(),seed=1)==generate_specs(config,fixture_plan(),seed=1)


def test_building_intersection_is_rejected():
    config=yaml.safe_load((ROOT/'config/logistics_city.yaml').read_text())
    bad=dict(center=[0,-12,9],amplitude=[1,1,.5],radius=.6)
    assert not orbit_is_safe(bad,config)


def test_delivery_world_keeps_city_and_excludes_only_visual_parcels_from_lidar(tmp_path):
    import xml.etree.ElementTree as ET
    source=ROOT/'simulation/worlds/logistics_city.sdf';before=source.read_bytes()
    out=tmp_path/'delivery.sdf';write_world(source,out,[])
    world=ET.parse(out).getroot().find('world')
    assert world.find("plugin[@name='mupt::PackageFollower']") is not None
    for i in range(1,7):
        assert world.find(f"model[@name='building_{i}']") is not None
        parcel=world.find(f"model[@name='package_{i}']")
        assert parcel is not None and parcel.findall('.//collision')==[]
        assert all(v.findtext('visibility_flags')=='2' for v in parcel.findall('.//visual'))
    rays=world.findall('.//sensor/ray')
    assert len(rays)==9
    assert all(int(r.findtext('visibility_mask'))&2==0 for r in rays)
    for i in range(1,4):
        sensor=world.find(f"model[@name='uav_{i}_lidar']/link/sensor[@name='safety_scan']")
        assert sensor.findtext('ray/scan/vertical/samples')=='11'
        trained=world.find(f"model[@name='uav_{i}_lidar']/link/sensor[@name='scan']")
        assert trained.findtext('ray/scan/vertical/samples')=='3'
    assert source.read_bytes()==before


def test_static_map_returns_do_not_trigger_dynamic_takeover():
    expected=np.full(36,5.,dtype=np.float32)
    assert novel_obstacle_minimum(expected,expected)==20.
    actual=expected.copy();actual[0]=4.6
    assert novel_obstacle_minimum(actual,expected)==20.
    actual[0]=3.
    assert novel_obstacle_minimum(actual,expected)==3.


def test_return_hands_back_to_astar_when_dynamic_obstacle_is_behind():
    static=np.full(36,20.);actual=static.copy();actual[12]=3.
    assert relevant_dynamic_minimum(actual,static,[-8,0,0],[-1.4,0,0])==20.
    assert relevant_dynamic_minimum(actual,static,[8,0,0],[1.4,0,0])==3.
    # A side obstacle approaching the UAV still triggers collision protection.
    actual=static.copy();before=static.copy();actual[15]=2.;before[15]=2.1
    assert relevant_dynamic_minimum(actual,static,[8,0,0],[1.4,0,0],before,.1,[0,0,0])==2.


def test_global_tracking_slowdown_preserves_direction_and_braking_margin():
    sectors=np.full(36,1.5)
    v=limit_tracking_speed([1.4,0,.2],sectors)
    assert v[0]>0 and v[2]/v[0]==pytest.approx(.2/1.4)
    _,braked=project_directional_velocity(v,sectors)
    assert not braked and np.linalg.norm(v)<1.4


def test_safety_projection_keeps_requested_clear_axes_without_new_velocity():
    sectors=np.full(36,20.);sectors[[0,12,24]]=.7
    requested=np.array([1.,.7,.3])
    selected,changed=project_requested_components(requested,sectors)
    assert changed and np.linalg.norm(selected)>0
    assert all(abs(v)<1e-6 or v==pytest.approx(r) for v,r in zip(selected,requested))
    stopped,_=project_requested_components(requested,np.full(36,.5))
    assert stopped==pytest.approx([0,0,0])
    clear,_=project_requested_components(requested,np.full(36,20.),lambda c:c[0]==0)
    assert clear==pytest.approx([0,.7,.3])


def test_radial_constraint_accounts_for_incoming_obstacle_and_ego_motion():
    expected=np.full(36,20.)
    measured=expected.copy();before=expected.copy()
    measured[12]=2.;before[12]=2.1
    n,b=dynamic_radial_bounds(measured,expected,before,.1,[.5,0,0])
    assert n.shape==(1,3) and n[0]==pytest.approx([1,0,0])
    assert b[0]==pytest.approx((2-1.4)/1.2-.5)
    projected,_=project_requested_components([1,.5,0],measured,radial_bounds=(n,b))
    assert projected[0]==0 and projected[1]==pytest.approx(.5)


def test_physical_safety_allows_recovery_from_voxel_inflation_without_crossing_building():
    city=CityPPOEnv(dynamic=False)
    config=load_world_config(ROOT/'config/logistics_city.yaml')
    grid=VoxelGrid(config)
    start=(6.4,-19.3,8.35);end=(5.2,-19.5,8.35)
    assert not grid.segment_is_free(start,end)
    assert physical_segment_free(start,end,city.boxes,city.tree_centres,config.minimum,config.maximum)
    assert not physical_segment_free(start,(0,-12,9),city.boxes,city.tree_centres,config.minimum,config.maximum)


def test_full_height_contract_keeps_legacy_default_and_allows_roof_routes():
    frames=np.full((4,36),20.)
    legacy=build_observation(frames,[0,0,1],[0,0,0],6,5)
    assert legacy[-3:-1]==pytest.approx([.5,.5])
    roof=build_observation(frames,[0,0,1],[0,0,0],25,2,altitude_max=32.)
    assert roof[-3:-1]==pytest.approx([23/30,7/30])
    velocity,_=project_safe_velocity([0,0,1],25,2,altitude_max=32.)
    assert velocity[2]==1


def test_delivery_env_observations_and_motion_are_full_height(tmp_path):
    config=yaml.safe_load((ROOT/'config/logistics_city.yaml').read_text())
    plan=fixture_plan()
    plan_path=tmp_path/'plan.json';plan_path.write_text(json.dumps(plan))
    manifest=tmp_path/'obstacles.json'
    manifest.write_text(json.dumps(dict(obstacles=generate_specs(config,plan,seed=1))))
    env=DeliveryPPOEnv(plan_path=plan_path,manifest_path=manifest,randomize=False)
    obs,_=env.reset(seed=2)
    assert env.observation_space.contains(obs) and obs.shape==(153,)
    assert env.corridor_max[2]==32 and len(env.obstacles)==17
    first=env.obstacles[-2].position.copy()
    env._place_obstacles(5)
    assert not np.allclose(first,env.obstacles[-2].position)
    for _ in range(10):
        obs,reward,terminated,truncated,info=env.step(np.zeros(3,dtype=np.float32))
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        if terminated or truncated:break


def test_delivery_plan_checks_order_capacity_and_complete_return():
    import copy
    plan=dict(tasks=[dict(id='task_1',quantity=1)],routes=[dict(capacity=1,delivery_legs=[
        dict(kind='pickup',task_id='task_1'),dict(kind='delivery',task_id='task_1'),dict(kind='return',task_id=None)])])
    validate_delivery_plan(plan)
    bad=copy.deepcopy(plan);bad['routes'][0]['delivery_legs'][0]['kind']='delivery'
    with pytest.raises(ValueError):validate_delivery_plan(bad)
    bad=copy.deepcopy(plan);bad['routes'][0]['capacity']=0
    with pytest.raises(ValueError):validate_delivery_plan(bad)
    bad=copy.deepcopy(plan);bad['routes'][0]['delivery_legs'].pop(1)
    with pytest.raises(ValueError):validate_delivery_plan(bad)


def test_geometry_audit_excludes_landing_and_rejects_cruise_contact(tmp_path):
    import csv
    (tmp_path/'obstacles.json').write_text(json.dumps(dict(obstacles=[dict(name='city_dynamic_1',radius=.6)])))
    with (tmp_path/'control_trace.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['vehicle','monotonic_s']);writer.writeheader()
        for i in (1,2,3):
            for t in (2,3):writer.writerow(dict(vehicle=i,monotonic_s=t))
    def frame(t,bad=False):
        positions=[[-34,-23,6],[-30,-23,10],[-26,-23,18]]
        if t==1:positions[0]=[-10,-25,4]  # pre-cruise crown contact must be excluded
        poses=[dict(name=f'x500_{i+1}',position=p) for i,p in enumerate(positions)]
        poses.append(dict(name='city_dynamic_1',position=[-25,-23,18] if bad else [32,26,30]))
        return dict(monotonic_s=t,poses=poses)
    path=tmp_path/'gazebo_poses.jsonl'
    path.write_text('\n'.join(json.dumps(frame(t)) for t in (1,2,3))+'\n')
    assert audit_geometry(tmp_path,ROOT/'config/logistics_city.yaml')['passed']
    path.write_text('\n'.join(json.dumps(frame(t,t==2)) for t in (1,2,3))+'\n')
    assert not audit_geometry(tmp_path,ROOT/'config/logistics_city.yaml')['passed']


def test_wide_cloud_stops_blind_descent_and_keeps_lateral_escape():
    from logistics_planner.delivery_control import cloud_radial_bounds,limit_cloud_velocity
    # Obstacle below the sensor, outside the original +/-12 degree view.
    points=np.array([[1.,0.,-1.8]])
    bounds=cloud_radial_bounds(points)
    v,changed=limit_cloud_velocity([1.,.5,-.8],bounds)
    assert changed and v[1]>.1
    assert np.all(bounds[0]@v<=bounds[1]+1e-6)
    # The same surface known to be static permits a controlled approach.
    static=cloud_radial_bounds(points,[True])
    assert static[1][0]>bounds[1][0]
    v,_=limit_cloud_velocity([-1.,0.,.5],bounds)
    assert v==pytest.approx([-1,0,.5])
    assert limit_cloud_velocity([1,0,0],cloud_radial_bounds(np.empty((0,3))))[0]==pytest.approx([1,0,0])


def test_balanced_destination_coverage_across_seeds():
    from dataclasses import replace
    from collections import Counter
    from logistics_planner.task_generator import generate_tasks
    config=load_world_config(ROOT/'config/logistics_city.yaml')
    for seed in range(30):
        tasks=generate_tasks(replace(config,task_settings=replace(config.task_settings,seed=seed)))
        counts=Counter(t.delivery.id for t in tasks)
        assert set(counts)=={d.id for d in config.deliveries}
        assert sorted(counts.values())==[2,2,2]
    with pytest.raises(ValueError,match='at least one'):
        generate_tasks(replace(config,task_settings=replace(config.task_settings,count=2)))


def test_delivery_plan_rejects_two_aircraft_serving_exclusive_destination():
    plan=dict(algorithm=dict(exclusive_delivery_uav=True),tasks=[
        dict(id='task_1',quantity=1,delivery='delivery_1'),
        dict(id='task_2',quantity=1,delivery='delivery_1')],routes=[])
    for i in (1,2):
        plan['routes'].append(dict(uav=f'uav_{i}',capacity=1,delivery_legs=[
            dict(kind='pickup',task_id=f'task_{i}'),dict(kind='delivery',task_id=f'task_{i}'),dict(kind='return',task_id=None)]))
    with pytest.raises(ValueError,match='multiple UAVs'):validate_delivery_plan(plan)


def test_return_keeps_climb_cruise_then_own_pad_descent():
    from logistics_planner.delivery_plan import add_delivery_legs
    plan=dict(algorithm={},tasks=[],routes=[dict(uav=f'uav_{i+1}',capacity=i+1,stops=[]) for i in range(3)])
    result=add_delivery_legs(plan,ROOT/'config/logistics_city.yaml')
    for i,route in enumerate(result['routes']):
        leg=route['delivery_legs'][-1];path=leg['path'];home=leg['target'];height=26.+2*i
        assert leg['kind']=='return' and leg['return_cruise_altitude_m']==height
        assert tuple(path[-1])==tuple(home)
        assert (home[0],home[1],height) in map(tuple,path)
        for first,last in zip(path,path[1:]):
            if last[2]<first[2]:
                assert last[:2]==pytest.approx(home[:2])
                assert first[:2]==pytest.approx(home[:2])


def test_roof_marker_with_sensor_noise_is_static_but_clear_orbit_is_unknown():
    from logistics_planner.delivery_control import known_static_cloud_mask
    boxes=np.array([[[-26,5,0],[-16,19,20]]],dtype=float)
    hits=np.array([[-21,10.5,20.24],[-21,10.5,20.27],[-21,10.5,20.55]])
    assert known_static_cloud_mask(hits,boxes,[]).tolist()==[True,True,False]


def test_sensed_blind_obstacle_replans_without_truth_and_leaves_base_grid_unchanged():
    from logistics_planner.delivery_control import plan_around_observed_points
    grid=VoxelGrid(load_world_config(ROOT/'config/logistics_city.yaml'))
    occupied=grid.occupied.copy()
    start=(-34.,-23.,26.);goal=(-24.,-23.,26.)
    point=np.array([[-29.,-23.,25.8]])
    result=plan_around_observed_points(grid,start,goal,point)
    assert grid.occupied==occupied
    assert result.points[0]==start and result.points[-1]==goal
    assert len(result.points)>2
    for a,b in zip(result.points,result.points[1:]):
        assert grid.segment_is_free(a,b)
        samples=np.linspace(a,b,30)
        assert np.min(np.linalg.norm(samples-point[0],axis=1))>1.


def test_priority_winner_cannot_drive_into_a_waiting_peer():
    from logistics_planner.delivery_control import fleet_radial_bounds,limit_cloud_velocity,plan_around_observed_points
    bounds=fleet_radial_bounds([[4.,0,0]],[[0,0,0]])
    command,changed=limit_cloud_velocity([1.4,0,0],bounds)
    assert changed and 0<=command[0]<=.4+1e-6
    # An approaching peer starts limiting earlier than a stationary peer.
    closing=fleet_radial_bounds([[6.,0,0]],[[-1.,0,0]])
    assert closing[1][0]<fleet_radial_bounds([[6.,0,0]],[[0,0,0]])[1][0]
    grid=VoxelGrid(load_world_config(ROOT/'config/logistics_city.yaml'))
    path=plan_around_observed_points(grid,(-34.,-23.,26.),(-24.,-23.,26.),[],peer_points=[[-29.,-23.,26.]])
    for a,b in zip(path.points,path.points[1:]):
        assert np.min(np.linalg.norm(np.linspace(a,b,30)-[-29,-23,26],axis=1))>3.
