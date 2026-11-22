"""Three concurrent A* pickup/delivery missions with direct PPO local velocities."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
import argparse
import csv
import json
import math
import time
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3Stamped
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus, VehicleLocalPosition
from rclpy.qos import qos_profile_sensor_data
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.scene_pb2 import Scene
from google.protobuf.text_format import MessageToString

from .offboard_mission import (MultiUAVOffboard, GazeboPackageManager, local_ned_to_enu,
    enu_vector_to_ned, ned_vector_to_body_flu, offset_point)
from .config import load_world_config
from .grid_planner import AStar3D, VoxelGrid
from .single_uav_avoidance_test import desired_velocity_enu, body_flu_to_ned, update_local_activation
from .direct_ppo_contract import project_safe_velocity, project_directional_velocity
from .delivery_plan import validate_delivery_plan
from .delivery_audit import audit_geometry
from .city_ppo_env import CityPPOEnv
from .delivery_control import relevant_dynamic_minimum,limit_tracking_speed,project_requested_components,dynamic_radial_bounds,physical_segment_free,cloud_radial_bounds,limit_cloud_velocity,known_static_cloud_mask,plan_around_observed_points,fleet_radial_bounds


class ConfirmedPackages(GazeboPackageManager):
    def native_update(self,request):
        # Keep synchronous transport bindings out of the ROS interpreter:
        # a pose subscription and a blocking request can otherwise starve it.
        service=self._service.rsplit('/',1)[0]+'/package_state'
        for _ in range(2):
            started=time.monotonic()
            try:
                result=subprocess.run(['gz','service','-s',service,'--reqtype','gz.msgs.Pose_V',
                    '--reptype','gz.msgs.Boolean','--timeout','1000','--req',MessageToString(request,as_one_line=True)],
                    capture_output=True,text=True,timeout=3)
                ok=result.returncode==0 and re.search(r'\bdata:\s*true\b',result.stdout) is not None
                record=dict(elapsed_s=time.monotonic()-started,success=ok,stdout=result.stdout[-400:],stderr=result.stderr[-400:])
            except subprocess.TimeoutExpired:
                ok=False;record=dict(elapsed_s=time.monotonic()-started,success=False,error='native package-state request timeout')
            if hasattr(self,'rpc_log'):
                with self.rpc_log.open('a') as stream:stream.write(json.dumps(record)+'\n')
            if ok:return True
        return False

    def carry_checked(self, task_ids, point, vehicle_index):
        request=Pose_V()
        for i,task in enumerate(task_ids):
            pose=request.pose.add()
            pose.name=f'package_{task.rsplit("_",1)[-1]}'
            pose.id=vehicle_index
            pose.position.x=point[0]+(i-(len(task_ids)-1)/2)*.45
            pose.position.y,pose.position.z=point[1],point[2]-.65
            pose.orientation.w=1.
        return self.native_update(request)

    def deliver_checked(self, task_id):
        task = self._tasks[task_id]
        peers = sorted(t['id'] for t in self._tasks.values() if t['delivery']==task['delivery'])
        point=offset_point(self._delivery_positions[task['delivery']], peers.index(task_id), len(peers))
        request=Pose_V();pose=request.pose.add()
        pose.name='package_'+task_id.rsplit('_',1)[-1]
        pose.id=0;pose.position.x,pose.position.y,pose.position.z=point;pose.orientation.w=1.
        return self.native_update(request)


class DeliveryPPOOffboard(MultiUAVOffboard):
    def __init__(self, plan_path, config_path, output):
        self.delivery_plan = json.loads(plan_path.read_text())
        validate_delivery_plan(self.delivery_plan)
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.delivered = set()
        self.service_owners={}
        self.data_hold_started=None
        self.communication_holds=[]
        self.latest_goals=[(0.,0.,0.) for _ in range(3)]
        self.ppo_ticks = [0, 0, 0]
        self.brake_ticks = [0, 0, 0]
        self.projected_ticks=[0,0,0]
        self.sensors = [dict(sectors=None, down=None, command=None, low=0) for _ in range(3)]
        self.flight = [dict(leg=0, waypoint=1, active=False, clear=0, active_since=0.,
                            dwell=0, last_progress=time.monotonic(), best_distance=math.inf,
                            replan_at=0., path=None) for _ in range(3)]
        self.started = time.monotonic()
        (self.output/'mission_clock.json').write_text(json.dumps(dict(monotonic_start=self.started))+'\n')
        self.workers=ThreadPoolExecutor(max_workers=3)
        self.payload_jobs=[None,None,None]
        config = load_world_config(config_path)
        self.grid = VoxelGrid(config)
        self.astar = AStar3D(self.grid)
        self.static_lidar=CityPPOEnv(dynamic=False,randomize=False)
        self.static_lidar.reset(seed=0)
        self.trace = (output / 'control_trace.csv').open('w', newline='', buffering=1)
        self.writer = csv.DictWriter(self.trace, fieldnames=['elapsed_s','vehicle','leg','kind','x','y','z',
            'source','minimum_scan','down','forward','left','up','reason','ppo_forward','ppo_left','ppo_up',
            'goal_forward','goal_left','goal_up','sectors_json','monotonic_s'])
        self.writer.writeheader()
        super().__init__(plan_path, config_path)
        self.package_manager = ConfirmedPackages(self.delivery_plan['world'], self.delivery_plan, config_path)
        self.package_manager.rpc_log=output/'package_rpc.jsonl'
        self.package_live=(0.,{})
        if not self.package_manager._node.subscribe(Pose_V,f'/world/{self.delivery_plan["world"]}/pose/info',self.package_pose_callback):
            raise RuntimeError('Could not subscribe to actual package poses')
        self.goal_publishers = []
        for index in range(3):
            root = f'/mupt/uav_{index+1}'
            self.goal_publishers.append(self.create_publisher(Vector3Stamped, root+'/planner/relative_goal_body', qos_profile_sensor_data))
            self.create_subscription(Float32MultiArray, root+'/observation/sectors',
                lambda m,i=index: self.observe(i,'sectors',list(m.data)), qos_profile_sensor_data)
            self.create_subscription(Float32, root+'/observation/down_clearance',
                lambda m,i=index: self.observe(i,'down',float(m.data)), qos_profile_sensor_data)
            self.create_subscription(PointCloud2,root+'/safety_scan/points',
                lambda m,i=index:self.observe_cloud(i,m),qos_profile_sensor_data)
            self.create_subscription(Vector3Stamped, root+'/ppo_direct/velocity_body',
                lambda m,i=index: self.observe(i,'command',(m.vector.x,m.vector.y,m.vector.z)), qos_profile_sensor_data)
            self.create_subscription(VehicleLocalPosition, f'/mupt_px4_{index+1}/fmu/out/vehicle_local_position_v1',
                lambda m,i=index: self.observe(i,'telemetry',0.), qos_profile_sensor_data)

    def observe(self, index, name, values):
        data = np.asarray(values)
        if not np.isfinite(data).all() or (name=='sectors' and data.shape!=(36,)):
            return
        if name=='sectors' and self.sensors[index].get(name) is not None:
            self.sensors[index]['previous_sectors']=self.sensors[index][name]
            self.sensors[index]['scan_dt']=time.monotonic()-self.sensors[index].get('sectors_time',time.monotonic())
        self.sensors[index][name] = values
        self.sensors[index][name+'_time'] = time.monotonic()

    def observe_cloud(self,index,message):
        cloud=point_cloud2.read_points(message,field_names=['x','y','z'],skip_nans=False)
        points=np.column_stack([cloud[k].reshape(-1) for k in ('x','y','z')])
        points=points[np.isfinite(points).all(axis=1)]
        self.sensors[index]['cloud']=points
        self.sensors[index]['cloud_time']=time.monotonic()

    def safety_cloud_bounds(self,index,current,heading):
        points=self.sensors[index]['cloud']
        # Identify only known static surfaces from the map; no simulator
        # moving-object poses are used for sensing or velocity decisions.
        yaw=math.pi/2-heading;c,s=math.cos(yaw),math.sin(yaw)
        world=points@np.array([[c,s,0.],[-s,c,0.],[0.,0.,1.]])+current+[0.,0.,.6]
        static=known_static_cloud_mask(world,self.static_lidar.boxes,self.static_lidar.tree_centres)
        relative=points+[0.,0.,.6]
        distance=np.linalg.norm(relative,axis=1)
        unknown=(~static)&(distance>.75)&(distance<5.)
        return cloud_radial_bounds(points,static),world[unknown],relative[unknown]

    def package_pose_callback(self,message):
        actual={p.name:(p.position.x,p.position.y,p.position.z) for p in message.pose
                if p.name.startswith('package_') and p.name[8:].isdigit()}
        if actual:self.package_live=(time.monotonic(),actual)

    def map_free(self, first, second):
        return physical_segment_free(first,second,self.static_lidar.boxes,self.static_lidar.tree_centres,self.grid.minimum,self.grid.maximum)

    def traffic_bounds(self,index,current,heading):
        relative=[];velocities=[];points=[]
        for j,m in enumerate(self.missions):
            if j==index or m.position is None or m.status is None or m.status.arming_state!=VehicleStatus.ARMING_STATE_ARMED:continue
            peer=np.array(local_ned_to_enu((m.position.x,m.position.y,m.position.z),m.spawn_enu))
            if np.linalg.norm(peer-current)>=8.:continue
            points.append(peer)
            relative.append(ned_vector_to_body_flu(enu_vector_to_ned(peer-current),heading))
            velocities.append(ned_vector_to_body_flu((m.position.vx,m.position.vy,m.position.vz),heading))
        return fleet_radial_bounds(relative,velocities),points

    def planning_segment_free(self,first,last):
        try:return self.grid.segment_is_free(tuple(first),tuple(last))
        except ValueError:return False

    def service_key(self,index):
        leg=self.delivery_plan['routes'][index]['delivery_legs'][self.flight[index]['leg']]
        return tuple(round(float(v),2) for v in leg['target']) if leg['kind']!='return' else None

    def _runtime_safety_holds(self):
        if not hasattr(self,'missions'):return set()
        positions={i:np.array(local_ned_to_enu((m.position.x,m.position.y,m.position.z),m.spawn_enu))
                   for i,m in enumerate(self.missions) if m.position is not None}
        for key,owner in list(self.service_owners.items()):
            if owner not in positions or np.linalg.norm(positions[owner]-key)>3.5 or self.missions[owner].landed:
                del self.service_owners[key]
        contenders=[]
        for i,m in enumerate(self.missions):
            if i not in positions or m.phase not in ('tracking','ppo_tracking'):continue
            key=self.service_key(i)
            if key is not None and key not in self.service_owners:
                distance=float(np.linalg.norm(positions[i]-key))
                if distance<3.5:contenders.append((distance,i,key))
        for _,i,key in sorted(contenders):
            self.service_owners.setdefault(key,i)
        holds=set()
        armed=[i for i,m in enumerate(self.missions) if i in positions and m.status is not None
               and m.status.arming_state==VehicleStatus.ARMING_STATE_ARMED]
        self.maximum_concurrent_airborne=max(self.maximum_concurrent_airborne,len(armed))
        owners=set(self.service_owners.values())
        for i in armed:
            key=self.service_key(i)
            if key is not None and key in self.service_owners and self.service_owners[key]!=i and np.linalg.norm(positions[i]-key)<6:
                holds.add(self.missions[i].namespace)
        for a,i in enumerate(armed):
            for j in armed[a+1:]:
                relative=positions[j]-positions[i]
                distance=float(np.linalg.norm(relative))
                self.minimum_actual_separation=min(self.minimum_actual_separation,distance)
                first,second=self.missions[i].position,self.missions[j].position
                velocity=np.array([second.vy-first.vy,second.vx-first.vx,-second.vz+first.vz])
                denom=float(velocity@velocity)
                t=float(np.clip(-(relative@velocity)/denom,0,1.5)) if denom>1e-6 else 0.
                if np.linalg.norm(relative+t*velocity)>=3.5:continue
                def priority(k):
                    m=self.missions[k]
                    return (0 if m.landing_commanded else 1 if k in owners else 2,m.trajectory_start_delay,m.system_id)
                loser=max((i,j),key=priority)
                holds.add(self.missions[loser].namespace)
        return holds

    def _update_package_state(self, mission):
        # Native Gazebo follower keeps cargo coherent without per-frame RPCs.
        pass

    def _publish_heartbeat(self, mission):
        message = OffboardControlMode()
        message.timestamp = self._timestamp()
        message.velocity = mission.phase in ('tracking','ppo_tracking')
        message.position = not message.velocity
        self.publishers_by_namespace[mission.namespace][0].publish(message)

    def _tick(self):
        try:
            stale=self.stale_inputs(time.monotonic())
            if stale:
                self.hold_for_data(stale)
                return
            if self.data_hold_started is not None:
                duration=time.monotonic()-self.data_hold_started
                self.communication_holds[-1]['duration_s']=duration
                for state in self.flight:state['last_progress']+=duration
                self.data_hold_started=None
                self.get_logger().info(f'Fresh data restored after {duration:.2f}s; resuming fleet')
            self._tick_impl()
        except Exception as error:
            self._abort_all(f'controller exception {type(error).__name__}: {error}')

    def stale_inputs(self,now):
        stale={}
        for i,m in enumerate(self.missions):
            if m.phase not in ('tracking','ppo_tracking') or m.landing_commanded:continue
            keys=['sectors','down','telemetry','cloud']+(['command'] if self.flight[i]['active'] else [])
            for key in keys:
                age=now-self.sensors[i].get(key+'_time',0.)
                if age>.5:stale[f'uav_{i+1}:{key}']=age
        return stale

    def hold_for_data(self,stale):
        now=time.monotonic()
        if self.data_hold_started is None:
            self.data_hold_started=now
            self.communication_holds.append(dict(elapsed_s=now-self.started,stale_ages_s=stale))
            self.get_logger().warning(f'Pausing fleet on stale observations: {stale}')
            for state in self.flight:state['dwell_since']=None
        for i,m in enumerate(self.missions):
            if m.phase in ('tracking','ppo_tracking') and not m.landing_commanded:
                goal=Vector3Stamped()
                goal.header.stamp=self.get_clock().now().to_msg()
                goal.vector.x,goal.vector.y,goal.vector.z=self.latest_goals[i]
                self.goal_publishers[i].publish(goal)
                self._publish_heartbeat(m)
                self._publish_setpoint(m,(math.nan,)*3,(0.,0.,0.),(math.nan,)*3)
        if now-self.data_hold_started>2.:
            self._abort_all(f'Sensor/telemetry outage exceeded two seconds: {stale}')

    def _tick_impl(self):
        if not hasattr(self,'missions'):
            return
        tracking = [m for m in self.missions if m.phase=='tracking' and not m.landing_commanded]
        for index, m in enumerate(self.missions):
            if m not in tracking and m.position is not None and not m.landing_commanded:
                current=local_ned_to_enu((m.position.x,m.position.y,m.position.z),m.spawn_enu)
                relative=np.array(m.staging_enu)-current
                msg=Vector3Stamped()
                msg.header.stamp=self.get_clock().now().to_msg()
                msg.vector.x,msg.vector.y,msg.vector.z=ned_vector_to_body_flu(enu_vector_to_ned(relative),m.position.heading)
                self.goal_publishers[index].publish(msg)
        super()._tick()

    def _custom_tracking_step(self,mission,now,must_hold):
        if self.data_hold_started is None and self.abort_reason is None:
            self.flight_step(self.missions.index(mission),mission,must_hold)
        return True

    def flight_step(self, index, mission, fleet_hold):
        """推进一个配送航段：感知校验、局部目标、PPO接管、安全约束和交接。

        位姿真值仅供离线核验；此处导航使用 PX4 遥测和实际激光观测。
        到达取送点后仍须低速驻留并确认货物服务，才能切换下一航段。
        """
        now = time.monotonic()
        data, state = self.sensors[index], self.flight[index]
        if any(now-data.get(key+'_time',0)>.5 for key in ('sectors','down','telemetry','cloud')):
            self.hold_for_data(self.stale_inputs(now))
            return
        current = np.array(local_ned_to_enu((mission.position.x,mission.position.y,mission.position.z),mission.spawn_enu))
        if current[2]<2 or current[2]>32 or np.any(current<self.grid.minimum) or np.any(current>self.grid.maximum):
            self._abort_all(f'{mission.namespace}: world or altitude bounds exceeded')
            return
        legs = self.delivery_plan['routes'][index]['delivery_legs']
        leg = legs[state['leg']]
        goal = np.array(leg['target'])
        distance = float(np.linalg.norm(goal-current))
        if state['path'] is None:
            state['path'] = leg['path']
            state['last_progress'] = now
            state['best_distance'] = distance
        if distance < state['best_distance']-.3:
            state['best_distance'], state['last_progress'] = distance, now
        if fleet_hold:
            state['last_progress'] = now
        elif now-state['last_progress']>75:
            self._abort_all(f'{mission.namespace}: no progress for 75 seconds')
            return
        path = state['path']
        future=state.get('replan_job')
        if future is not None and future.done():
            try:
                state['path']=future.result().points
                state['path'].extend(state.pop('replan_tail',[]))
                state['waypoint']=1
                path=state['path']
            except (ValueError,RuntimeError):pass
            state['replan_job']=None
        while state['waypoint']<len(path)-1 and math.dist(current,path[state['waypoint']])<.9:
            state['waypoint']+=1
        target = np.array(path[min(state['waypoint'],len(path)-1)])
        if not self.planning_segment_free(current,target) and now-state['replan_at']>2:
            state['replan_at'] = now
            if state.get('replan_job') is None:
                # A return replan must retain climb/cruise/pad-descent gates.
                replan_goal=target if leg['kind']=='return' else goal
                state['replan_tail']=path[state['waypoint']+1:] if leg['kind']=='return' else []
                state['replan_job']=self.workers.submit(self.astar.plan,tuple(current),tuple(replan_goal))
        delta = target-current
        target = current + delta*min(1.,8./max(float(np.linalg.norm(delta)),1e-6))
        relative_body = ned_vector_to_body_flu(enu_vector_to_ned(target-current),mission.position.heading)
        self.latest_goals[index]=relative_body
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'uav_{index+1}/lidar_link'
        msg.vector.x,msg.vector.y,msg.vector.z = relative_body
        self.goal_publishers[index].publish(msg)
        minimum = min(data['sectors'])
        self.static_lidar.position=current.astype(np.float32)
        self.static_lidar.heading=math.pi/2-mission.position.heading
        expected=self.static_lidar._scan()
        world=np.array(desired_velocity_enu(current,goal if distance<1.8 else target))
        nominal=ned_vector_to_body_flu(enu_vector_to_ned(world),mission.position.heading)
        cloud_bounds,observed_points,relative_points=self.safety_cloud_bounds(index,current,mission.position.heading)
        traffic_bounds,traffic_points=self.traffic_bounds(index,current,mission.position.heading)
        traffic_risk=bool(len(traffic_bounds[1]) and np.any(traffic_bounds[0]@np.asarray(nominal)>traffic_bounds[1]))
        wide_risk=False
        if len(relative_points):
            ranges=np.linalg.norm(relative_points,axis=1)
            toward=(relative_points@np.asarray(nominal))/ranges
            wide_risk=bool(np.any(toward>np.maximum(0.,(ranges-1.5)/1.8-.35)) or ranges.min()<3.)
        if (wide_risk or traffic_risk) and now-state.get('dynamic_replan_at',0.)>1.5 and state.get('replan_job') is None:
            state['dynamic_replan_at']=now
            replan_goal=target if leg['kind']=='return' else goal
            state['replan_tail']=path[state['waypoint']+1:] if leg['kind']=='return' else []
            state['replan_job']=self.workers.submit(plan_around_observed_points,self.grid,tuple(current),tuple(replan_goal),observed_points.copy(),1.3,traffic_points)
        body_velocity=ned_vector_to_body_flu((mission.position.vx,mission.position.vy,mission.position.vz),mission.position.heading)
        novelty=relevant_dynamic_minimum(data['sectors'],expected,relative_body,nominal,
            data.get('previous_sectors'),data.get('scan_dt',.1),body_velocity)
        if wide_risk:novelty=min(novelty,3.)
        was_active = state['active']
        state['active'],state['clear'] = update_local_activation(state['active'],state['clear'],novelty,now-state['active_since'])
        if state['active'] and not was_active:
            state['active_since'] = now
        source,reason = 'astar','waypoint_tracking'
        command=nominal
        if state['active'] and (distance>=1.8 or wide_risk or novelty<3.):
            if data['command'] is None or now-data.get('command_time',0)>.5:
                self.hold_for_data({f'uav_{index+1}:command':now-data.get('command_time',0.)})
                return
            command = data['command']
            source,reason = 'ppo','direct_3d'
        if source=='astar':command=limit_tracking_speed(command,data['sectors'])
        command,flags = project_safe_velocity(command,float(current[2]),data['down'],altitude_min=2.,altitude_max=32.)
        def candidate_is_free(candidate):
            ned=body_flu_to_ned(candidate,mission.position.heading)
            return self.map_free(current,current+np.array([ned[1],ned[0],-ned[2]])*.8)
        bounds=dynamic_radial_bounds(data['sectors'],expected,data.get('previous_sectors'),data.get('scan_dt',.1),body_velocity)
        command,projected=project_requested_components(command,data['sectors'],candidate_is_free,bounds)
        combined=(np.vstack([cloud_bounds[0],traffic_bounds[0]]),np.concatenate([cloud_bounds[1],traffic_bounds[1]]))
        command,cloud_projected=limit_cloud_velocity(command,combined)
        projected=projected or cloud_projected
        braked=projected and np.linalg.norm(command)<.01
        if projected:
            self.projected_ticks[index]+=1
            reason+=':requested_components_projected'
        data['low'] = data['low']+1 if minimum<.8 else 0
        if fleet_hold or braked:
            command = np.zeros(3)
            source = 'hold' if fleet_hold else 'brake'
            self.brake_ticks[index]+=1
        elif source=='ppo':
            self.ppo_ticks[index]+=1
        raw=data['command'] or (math.nan,math.nan,math.nan)
        self.writer.writerow(dict(elapsed_s=round(now-self.started,3),vehicle=index+1,leg=state['leg'],kind=leg['kind'],
            x=current[0],y=current[1],z=current[2],source=source,minimum_scan=minimum,down=data['down'],
            forward=command[0],left=command[1],up=command[2],reason=reason+str(flags),
            ppo_forward=raw[0],ppo_left=raw[1],ppo_up=raw[2],goal_forward=relative_body[0],goal_left=relative_body[1],goal_up=relative_body[2],
            sectors_json=json.dumps(data['sectors']),monotonic_s=now))
        if data['low']>=2:
            self._abort_all(f'{mission.namespace}: consecutive scan distance below .8 m')
            return
        self._publish_setpoint(mission,(math.nan,)*3,body_flu_to_ned(command,mission.position.heading),(math.nan,)*3)
        if fleet_hold or not self._settled_at(mission,tuple(goal),radius=.8,maximum_speed=.45):
            state['dwell']=0
            state['dwell_since']=None
            return
        if state.get('dwell_since') is None:
            state['dwell_since']=now
        if now-state['dwell_since']<2.:
            return
        task = leg['task_id']
        following=self.payload_jobs[index]
        if following is not None and not following.done():return
        if leg['kind']=='pickup':
            quantity = {t['id']:t['quantity'] for t in self.delivery_plan['tasks']}
            capacity = self.delivery_plan['routes'][index]['capacity']
            if sum(quantity[t] for t in mission.onboard_tasks)+quantity[task]>capacity:
                self._abort_all('payload capacity exceeded')
                return
            job=state.get('service_job')
            if job is None:
                state['service_job']=self.workers.submit(self.package_manager.carry_checked,mission.onboard_tasks+[task],tuple(current),index+1)
                return
            if not job.done():return
            if not job.result():
                self._abort_all('Gazebo refused package pickup pose update')
                return
            if task not in mission.onboard_tasks:mission.onboard_tasks.append(task)
        elif leg['kind']=='delivery':
            if task not in mission.onboard_tasks:
                self._abort_all('delivery attempted without pickup')
                return
            job=state.get('service_job')
            if job is None:
                state['service_job']=self.workers.submit(self.package_manager.deliver_checked,task)
                return
            if not job.done():return
            if not job.result():
                self._abort_all('Gazebo refused package delivery update')
                return
            mission.onboard_tasks.remove(task)
            self.delivered.add(task)
        self.events.append(dict(vehicle=index+1,kind=leg['kind'],task=task,elapsed_s=now-self.started))
        (self.output/'package_events.json').write_text(json.dumps(self.events,indent=2)+'\n')
        self.get_logger().info(f'uav_{index+1}: {leg["kind"]} {task}')
        if leg['kind']=='return':
            mission.phase='landing'
            mission.landing_commanded=True
            self._vehicle_command(mission,VehicleCommand.VEHICLE_CMD_NAV_LAND)
        else:
            pending=state.get('replan_job')
            if pending is not None:pending.cancel()
            state.update(leg=state['leg']+1,waypoint=1,path=None,dwell=0,dwell_since=None,best_distance=math.inf,last_progress=now,
                         service_job=None,replan_job=None)

    def save_summary(self):
        self.trace.flush()
        geometry=audit_geometry(self.output,self.output/'city_config.yaml')
        verified={}
        stamp,actual=self.package_live
        if time.monotonic()-stamp>1.:actual={}
        for task in self.delivery_plan['tasks']:
            peers=sorted(t['id'] for t in self.delivery_plan['tasks'] if t['delivery']==task['delivery'])
            target=offset_point(self.package_manager._delivery_positions[task['delivery']],peers.index(task['id']),len(peers))
            name='package_'+task['id'].rsplit('_',1)[-1]
            verified[task['id']]=name in actual and math.dist(actual[name],target)<.2
        returned=[any(e['vehicle']==i+1 and e['kind']=='return' for e in self.events) for i in range(3)]
        home_error=[math.dist(local_ned_to_enu((m.position.x,m.position.y,m.position.z),m.spawn_enu)[:2],m.spawn_enu[:2])
                    if m.position is not None else None for m in self.missions]
        destination_counts={d:sum(t['id'] in self.delivered and t['delivery']==d for t in self.delivery_plan['tasks'])
                            for d in self.package_manager._delivery_positions}
        coverage=all(n>0 for n in destination_counts.values()) if self.delivery_plan['algorithm'].get('task_distribution')=='balanced' else True
        success = self.abort_reason is None and self.complete and all(returned) and all(e is not None and e<1.5 for e in home_error) and coverage and len(self.delivered)==len(self.delivery_plan['tasks']) and self.minimum_actual_separation>=2.5 and all(verified.values()) and self.maximum_concurrent_airborne>=3 and geometry['passed']
        if not success and self.abort_reason is None:
            self.abort_reason='Final verification failed: incomplete task/landing, package pose, fleet separation, or concurrent-flight criterion'
        result = dict(success=success,abort_reason=self.abort_reason,
            delivered=sorted(self.delivered),expected_tasks=len(self.delivery_plan['tasks']),
            landed=[m.landed for m in self.missions],max_concurrent_airborne=self.maximum_concurrent_airborne,
            returned_home=returned,landing_home_xy_error_m=home_error,delivered_by_destination=destination_counts,
            minimum_actual_separation=self.minimum_actual_separation,ppo_ticks=self.ppo_ticks,
            brake_ticks=self.brake_ticks,potential_field_used=False,
            projected_ticks=self.projected_ticks,
            geometry_audit=geometry,
            communication_holds=self.communication_holds,
            onboard_tasks=[m.onboard_tasks for m in self.missions],package_destination_verified=verified,
            payload_mode='visual packages follow UAV; mass and cable dynamics not simulated')
        (self.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        (self.output/'package_events.json').write_text(json.dumps(self.events,indent=2)+'\n')
        self.trace.close()
        return success


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--timeout',type=float,default=900)
    args=parser.parse_args()
    rclpy.init()
    node=DeliveryPPOOffboard(args.plan,args.config,args.output_dir)
    success=False
    try:
        deadline=time.monotonic()+args.timeout
        while rclpy.ok() and not node.complete and time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.1)
        if not node.complete:
            node._abort_all('delivery mission timeout')
            deadline=time.monotonic()+60
            while not node.complete and time.monotonic()<deadline:
                rclpy.spin_once(node,timeout_sec=.1)
        success=node.save_summary()
        print('PASS: three concurrent deliveries complete' if success else f'FAIL: {node.abort_reason}',flush=True)
    finally:
        # Stop native callbacks while Python and the ROS node are still alive.
        # Otherwise transport may enter a callback during interpreter teardown.
        node.package_manager._node.unsubscribe(f'/world/{node.delivery_plan["world"]}/pose/info')
        node.workers.shutdown(wait=True,cancel_futures=True)
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__=='__main__':
    main()
