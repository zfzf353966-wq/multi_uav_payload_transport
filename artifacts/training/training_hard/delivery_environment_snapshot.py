"""Full-height city A* leg training with matching direct-PPO runtime guards."""
import json
import math
from collections import deque
from pathlib import Path
import numpy as np
import gymnasium as gym
import yaml
from .city_ppo_env import CityPPOEnv
from .city_dynamic_obstacles import generate_specs, orbit_position
from .direct_ppo_contract import action_to_velocity, build_observation, project_safe_velocity, project_directional_velocity
from .grid_planner import VoxelGrid, AStar3D
from .config import load_world_config
from .rl_avoidance_env import MovingSphere


class DeliveryPPOEnv(CityPPOEnv):
    def __init__(self, *, plan_path, manifest_path, dynamic=True, randomize=True,
                 reward_revision=2, static_probability=0., policy_only=False,
                 departure_probability=0., replay_probability=0., **unused):
        super().__init__(dynamic=dynamic, randomize=randomize, max_steps=750)
        self.plan = json.loads(Path(plan_path).read_text())
        self.base_specs = json.loads(Path(manifest_path).read_text())['obstacles']
        root = Path(__file__).resolve().parents[3]
        self.raw_config = yaml.safe_load((root/'config/logistics_city.yaml').read_text())
        self.grid = VoxelGrid(load_world_config(root/'config/logistics_city.yaml'))
        self.astar = AStar3D(self.grid)
        self.legs = [leg for route in self.plan['routes'] for leg in route['delivery_legs']
                     if len(leg['path'])>1 and math.dist(leg['path'][0],leg['path'][-1])>1]
        self.static_probability=static_probability
        self.policy_only=policy_only
        self.departure_probability=departure_probability
        self.replay_probability=replay_probability
        self.departure_legs=[r['delivery_legs'][0] for r in self.plan['routes']]
        self.corridor_min=np.array([-39.,-29.,2.],dtype=np.float32)
        self.corridor_max=np.array([39.,29.,32.],dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        pool=self.departure_legs if self.departure_probability and self.np_random.random()<self.departure_probability else self.legs
        leg=pool[int(self.np_random.integers(len(pool)))]
        self.path=[np.array(p,dtype=np.float32) for p in leg['path']]
        self.start=self.path[0].copy()
        self.final_goal=self.path[-1].copy()
        self.position=self.start.copy()
        self.velocity=np.zeros(3,dtype=np.float32)
        self.heading=float(self.np_random.uniform(-.1,.1)) if self.randomize else 0.
        self.previous_action=np.zeros(3,dtype=np.float32)
        self.step_count=0
        self.sweep_start=None
        self.waypoint=1
        self.local_active=False
        self.local_clear_ticks=0
        self.local_started=0
        self.low_scan_streak=0
        self.last_ttc=math.inf
        self.outbound_goal=self.final_goal.copy()
        self.outbound_complete=False
        self.orbits=[]
        self.specs=[]
        with_dynamic=self.dynamic and (not self.static_probability or self.np_random.random()>=self.static_probability)
        if with_dynamic:
            if self.randomize:
                scene_seed=1 if self.replay_probability and self.np_random.random()<self.replay_probability else int(self.np_random.integers(10000000))
                self.specs=generate_specs(self.raw_config,self.plan,seed=scene_seed)
            else:
                self.specs=self.base_specs
            self.specs=list(self.specs)+[dict(name='moving_obstacle_1',center=[10.,0.,8.],radius=.45,
                amplitude=[4.,4.,0.],rate=[.5,.5,0.],phase=[0.,0.,0.])]
        self.obstacles=[MovingSphere(c.copy(),np.zeros(3,dtype=np.float32),1.55) for c in self.tree_centres]
        self.obstacles += [MovingSphere(np.array(s['center'],dtype=np.float32),np.zeros(3,dtype=np.float32),s['radius']) for s in self.specs]
        self.time_offset=float(self.np_random.uniform(0,60)) if self.randomize else 0.
        self.response_time=float(self.np_random.uniform(.3,.55)) if self.randomize else .35
        self.maximum_acceleration=float(self.np_random.uniform(2.,3.2)) if self.randomize else 3.
        self.delay=int(self.np_random.integers(0,3)) if self.randomize else 0
        self.queue=deque(np.zeros(3,dtype=np.float32) for _ in range(self.delay))
        self.scan_history.clear()
        for i in range(4):
            self._place_obstacles((i-3)*self.dt)
            self.scan_history.append(self._scan())
        return self._observation(),self.info(False,False,False,False,False)

    def _place_obstacles(self,time_seconds=None):
        t=(self.step_count*self.dt if time_seconds is None else time_seconds)+self.time_offset
        for obstacle,spec in zip(self.obstacles[len(self.tree_centres):],self.specs):
            obstacle.position=orbit_position(spec,t).astype(np.float32)
            a,w,p=(np.array(spec[k]) for k in ('amplitude','rate','phase'))
            obstacle.velocity=(a*w*np.array([-np.sin(w[0]*t+p[0]),np.cos(w[1]*t+p[1]),np.cos(w[2]*t+p[2])])).astype(np.float32)

    def target(self):
        while self.waypoint<len(self.path)-1 and np.linalg.norm(self.position-self.path[self.waypoint])<.9:
            self.waypoint+=1
        delta=self.path[self.waypoint]-self.position
        return self.position+delta*min(1.,8./max(float(np.linalg.norm(delta)),1e-6))

    def remaining(self):
        return float(np.linalg.norm(self.position-self.path[self.waypoint]))+sum(
            float(np.linalg.norm(b-a)) for a,b in zip(self.path[self.waypoint:],self.path[self.waypoint+1:]))

    def _observation(self):
        goal=self._world_to_body(self.target()-self.position)
        return build_observation(tuple(self.scan_history),goal,self._world_to_body(self.velocity),float(self.position[2]),
                                 self._down_clearance(),altitude_min=2.,altitude_max=32.)

    def info(self,collision,success,outside,braked,ppo):
        return dict(collision=collision,success=success,outside=outside,direction_braked=braked,
                    minimum_clearance=self._clearance(),safety_guarded=braked,ppo_control=ppo)

    def step(self,action):
        from .single_uav_avoidance_test import update_local_activation,desired_velocity_enu
        target=self.target()
        previous_remaining=self.remaining()
        if self.step_count%20==0 and not self.grid.segment_is_free(tuple(self.position),tuple(target)):
            try:
                self.path=[np.array(p,dtype=np.float32) for p in self.astar.plan(tuple(self.position),tuple(self.final_goal)).points]
                self.waypoint=1
                target=self.target()
                previous_remaining=self.remaining()
            except (ValueError,RuntimeError):
                pass
        old=self.local_active
        self.local_active,self.local_clear_ticks=update_local_activation(self.local_active,self.local_clear_ticks,
            float(self.scan_history[-1].min()),(self.step_count-self.local_started)*self.dt)
        if self.local_active and not old:self.local_started=self.step_count
        ppo=self.local_active and np.linalg.norm(self.final_goal-self.position)>=1.8
        if self.policy_only:ppo=True
        if ppo:
            command=action_to_velocity(action)
        else:
            waypoint=self.final_goal if np.linalg.norm(self.final_goal-self.position)<1.8 else target
            command=self._world_to_body(np.array(desired_velocity_enu(self.position,waypoint),dtype=np.float32))
        self.queue.append(command)
        command=self.queue.popleft()
        command,braked=project_directional_velocity(command,self.scan_history[-1])
        command,flags=project_safe_velocity(command,float(self.position[2]),self._down_clearance(),altitude_min=2.,altitude_max=32.)
        world=self._body_to_world(command)
        if not self.grid.segment_is_free(tuple(self.position),tuple(self.position+world*.8)):
            world=np.zeros(3,dtype=np.float32)
            braked=True
        acceleration=(world-self.velocity)/self.response_time
        norm=float(np.linalg.norm(acceleration))
        acceleration*=min(1.,self.maximum_acceleration/max(norm,1e-9))
        self.sweep_start=self.position.copy()
        old_obstacles=[o.position.copy() for o in self.obstacles]
        self.velocity+=acceleration*self.dt
        self.position+=self.velocity*self.dt
        self.step_count+=1
        self._place_obstacles()
        clearance=self._clearance()
        for old_obstacle,obstacle in zip(old_obstacles,self.obstacles):
            relative=old_obstacle-self.sweep_start
            delta=obstacle.position-self.position-relative
            fraction=float(np.clip(-(relative@delta)/max(float(delta@delta),1e-12),0,1))
            clearance=min(clearance,float(np.linalg.norm(relative+fraction*delta))-obstacle.radius-self.agent_radius)
        self.sweep_start=None
        self.scan_history.append(self._scan())
        self.target()
        progress=previous_remaining-self.remaining()
        collision=clearance<=0
        outside=bool(np.any(self.position<self.corridor_min) or np.any(self.position>self.corridor_max))
        success=np.linalg.norm(self.final_goal-self.position)<.8 and not collision and not outside
        self.low_scan_streak=self.low_scan_streak+1 if self.scan_history[-1].min()<.8 else 0
        reward=12.*progress-.04-.005*self.remaining()-5.*max(0.,1.2-clearance)**2
        if braked and ppo:reward-=.8
        if np.linalg.norm(self.velocity)<.15:reward-=.2
        if collision or outside:reward-=1000.
        elif success:reward+=150.
        terminated=bool(collision or outside or success)
        truncated=bool(not terminated and (self.low_scan_streak>=2 or self.step_count>=self.max_steps))
        if truncated:reward-=300.
        return self._observation(),float(reward),terminated,truncated,self.info(collision,bool(success),outside,braked,ppo)
