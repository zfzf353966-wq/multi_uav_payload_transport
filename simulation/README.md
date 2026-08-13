# Logistics City simulation

`worlds/logistics_city.sdf` is a lightweight Gazebo Harmonic world for the
three-UAV delivery scenario. It uses only primitive geometry and two local
models, so it does not require Gazebo Fuel downloads.

Scene legend:

- cyan circles: UAV launch pads at the depot;
- yellow squares: rooftop pickup points;
- blue circles: rooftop delivery points;
- orange cubes: six task packages (`package_1` through `package_6`);
- red moving sphere: dynamic obstacle used as a bird substitute;
- trees and buildings: collision-enabled static obstacles.

Each X500 also has a visual-free 360-degree, three-layer range sensor. Every
sensor uses 120 horizontal samples at elevation angles -12, 0, and +12 degrees
(360 rays total) at 10 Hz with a 1--20 m range. The 1 m blind radius excludes
the X500 airframe from its own scan. The three scan topics are
`/mupt/uav_1/scan`, `/mupt/uav_2/scan`, and `/mupt/uav_3/scan`. A lightweight
pose follower keeps the sensors aligned with their X500 models. Gazebo also
publishes a compact point cloud below each scan topic as `/points`; no camera
or dense 3D lidar is used.

Coordinates are in Gazebo ENU. The matching planner/task configuration is
`../config/logistics_city.yaml`.

Run the world alone:

```bash
source scripts/setup_env.bash
gz sim -r simulation/worlds/logistics_city.sdf
```

Run the complete three-PX4 simulation:

```bash
scripts/run_three_px4.bash --gui
```

In a second terminal, generate and execute the traditional baseline:

```bash
source scripts/setup_env.bash
ros2 run logistics_planner traditional_baseline \
  --output-dir runs/traditional_baseline_ros
ros2 run logistics_planner offboard_mission \
  --plan runs/traditional_baseline_ros/plan.json
```

The baseline executor runs all three assigned routes concurrently. A
prioritized space-time reservation table covers staging waits, continuous
flight, and a 12-second landing reservation. The three staging altitudes are
3, 10.5, and 18 m, and the launch pads are 4 m apart. Runtime protection uses
actual PX4 positions and projected relative motion to pause the shared virtual
trajectory clock if separation is threatened. A package is reset to its
planned pickup roof when the mission starts, follows its assigned UAV after
pickup, and remains on the destination roof after delivery. Packages are
visual-only in this phase, so their mass does not yet alter vehicle dynamics.

PX4 receives time-parameterized position, velocity, and acceleration setpoints
at 10 Hz. A rounded, collision-checked path and curvature-aware speed profile
limit stop-and-go motion and slow the UAV before bends. The vehicle only
performs a position-and-speed settle at its final staging point before
commanding land. This is closed-loop control using PX4's local-position
estimate; minimum-snap optimization can be added as a later refinement.

Run the headless integration check and exit automatically:

```bash
scripts/run_three_px4.bash --check
```

## Lidar and RViz

After the three-PX4 simulation is running, open a second terminal:

```bash
cd /home/gbc/projects/multi_uav_payload_transport
source scripts/setup_env.bash
scripts/run_lidar_rviz.bash
```

This starts the Gazebo-to-ROS bridge, publishes `map` transforms for all three
sensor frames, and opens RViz. The three layered point clouds are red, green,
and blue. To
bridge the data without opening RViz, use:

```bash
scripts/run_lidar_rviz.bash --no-rviz
```

Check a scan from another sourced terminal:

```bash
ros2 topic hz /mupt/uav_1/scan
ros2 topic echo --once /mupt/uav_1/scan \
  --qos-reliability best_effort
```

The middle layer remains compatible with the earlier horizontal scan use case.
The upper and lower layers add lightweight vertical coverage for obstacles
approaching from a different altitude. RViz uses the `/points` topics so all
three elevation layers retain their true 3D positions.

## Sector observations

The lidar launcher also compresses every organized point cloud into 36 nearest
distances on `/mupt/uav_N/observation/sectors`. The fixed layout is:

```text
[lower 12 sectors, middle 12 sectors, upper 12 sectors]
```

Within each layer, sector 0 is centred on the vehicle's forward `+X` axis and
the remaining sectors advance counter-clockwise in 30-degree steps. Values are
metres in the range 1--20; a value of 20 means no obstacle was detected in
that sector. This representation can later be normalized by dividing by 20
before it is passed to a reinforcement-learning policy.

Check one observation:

```bash
ros2 topic echo --once /mupt/uav_1/observation/sectors \
  std_msgs/msg/Float32MultiArray --qos-reliability best_effort
```

## Shadow avoidance

`scripts/run_lidar_rviz.bash` also starts an advisory-only local avoidance
node. It reads the 36 sector distances and the traditional mission executor's
10 Hz body-frame desired velocity. A three-frame median filter rejects
isolated lidar spikes before a potential-field rule computes a body-frame
velocity correction. It publishes diagnostics only and never writes to a PX4
input topic:

```text
/mupt/uav_N/avoidance/action
/mupt/uav_N/avoidance/correction_body
/mupt/uav_N/avoidance/suggested_velocity_body
/mupt/uav_N/planner/desired_velocity_body
```

The vector frame is forward-left-up. `CLEAR` means no relevant obstacle is
within 6 m. `AVOID_*` and `EMERGENCY_*` report the suggested direction and the
nearest relevant distance. This node is a communication and rule-baseline
check before local avoidance is allowed to modify the real flight setpoint.

## Single-UAV closed-loop avoidance test

The isolated test promotes the advisory velocity to a real PX4 velocity
setpoint for UAV 1 only. UAV 2 and UAV 3 remain disarmed. The wrapper creates a
temporary red sphere at `(-20, -23, 6)` directly on a low-speed test corridor,
then removes it automatically when the test exits. Do not run
`offboard_mission` at the same time because two nodes must never command the
same PX4 instance.

Start the simulation and lidar stack in two terminals:

```bash
scripts/run_three_px4.bash --gui
```

```bash
scripts/run_lidar_rviz.bash
```

Run the isolated controller in a third terminal:

```bash
scripts/run_single_avoidance_test.bash
```

The controller uses a 1.4 m/s nominal speed and a 2.0 m/s hard command limit.
It aborts and lands on stale avoidance data, PX4 failure, Offboard loss,
geofence violation, or timeout. Success requires the vehicle to pass the
obstacle, produce a measurable lateral detour, return to the launch area, land,
and disarm. The final line is similar to:

```text
PASS: UAV 1 static closed-loop avoidance completed; avoidance_ticks=..., minimum_scan=... m, lateral_detour=... m
```

### Moving-obstacle closed-loop test

After the static test passes, keep the same simulation and lidar terminals
running and use the dynamic wrapper instead:

```bash
scripts/run_single_dynamic_avoidance_test.bash
```

This creates a 0.75 m radius red sphere whose centre follows a 2.5 m circular
orbit at about 0.88 m/s. The orbit is centred at `(-20, -23, 6)` and repeatedly
crosses UAV 1's nominal route. The temporary model is removed automatically.
A successful run ends with:

```text
PASS: UAV 1 dynamic closed-loop avoidance completed; avoidance_ticks=..., minimum_scan=... m, lateral_detour=... m
```

## Lightweight reinforcement-learning environment

`UAVLocalAvoidanceEnv` is a fast Gymnasium environment used before expensive
Gazebo/PX4 training. Its 150-value observation contains four consecutive
normalized 3x12 lidar frames, relative goal position, and vehicle velocity.
The four frames cover about 0.3 seconds and expose obstacle approach trends.
The three continuous actions are bounded forward-left-up residual corrections
on top of the artificial-potential-field suggestion. Route
direction, vehicle heading, and five moving spheres are randomized at every
reset. Each episode requires an outbound leg and a return to the start;
rewards cover goal progress, clearance, smooth control, collision, geofence
exit, the outbound waypoint, and successful return.

The reward also penalizes predicted time-to-collision below six seconds,
approaching the geofence before an actual violation, and excessive departure
from the global route. A zero RL action exactly preserves the potential-field
baseline, so PPO only has to learn when and how to improve that controller.

Run a short end-to-end PPO smoke test without starting ROS or Gazebo:

```bash
scripts/train_rl_avoidance.bash \
  --timesteps 4096 \
  --eval-episodes 30 \
  --output-dir runs/rl_avoidance_smoke
```

The command saves `ppo_local_avoidance.zip`, TensorBoard data, and
`metrics.json`. Metrics always compare the trained policy against the
zero-correction nominal controller on identical episode seeds. A smoke run
only validates sampling, optimization, serialization, and evaluation; it is
not accepted as a deployable policy unless it improves collision and success
rates over the nominal baseline on a much larger evaluation set.

After the smoke test, run the first meaningful training gate with parallel CPU
environments and a fixed 200-episode holdout set:

```bash
scripts/train_rl_avoidance.bash \
  --timesteps 100000 \
  --eval-episodes 200 \
  --n-envs 8 \
  --output-dir runs/rl_avoidance_100k
```

The evaluation reports success, collision, geofence-exit, and timeout rates.
Passing this lightweight gate only permits a policy to enter advisory Gazebo
inference. It must still pass Gazebo shadow comparison and the existing
single-UAV safety controller before it is allowed to command PX4.

The earlier direct-correction round-trip policy was trained with:

```bash
scripts/train_rl_avoidance.bash \
  --timesteps 300000 \
  --eval-episodes 300 \
  --n-envs 8 \
  --output-dir runs/rl_round_trip_300k
```

Its fixed-seed holdout result was 64% complete round trips, 12.7% collisions,
22.3% geofence exits, and 86% outbound completion. The model is stored under
`runs/`, which is intentionally excluded from Git; train it locally before
starting the PPO shadow process.

The current four-frame residual policy is trained with:

```bash
scripts/train_rl_avoidance.bash \
  --timesteps 300000 \
  --eval-episodes 300 \
  --n-envs 8 \
  --output-dir runs/rl_residual_stack4
```

On the fixed 300-episode lightweight holdout, the potential-field baseline
achieved 91.33% success, 1.33% collision, and 7.33% timeout. The residual PPO
achieved 97.33% success, 0.33% collision, 1.67% geofence exit, and 0.67%
timeout. Gazebo safety arbitration and the hard geofence therefore remain
mandatory despite the higher lightweight success rate.

### PPO shadow inference in Gazebo

The trained policy can consume live Gazebo observations at 10 Hz without
writing any PX4 input topic. Start the normal simulation, lidar stack, and PPO
shadow process in separate terminals:

```bash
scripts/run_three_px4.bash --gui
```

```bash
scripts/run_lidar_rviz.bash
```

```bash
scripts/run_rl_shadow.bash
```

Then start the existing dynamic test in a fourth terminal. The artificial
potential-field controller remains responsible for the actual flight; PPO is
advisory only:

```bash
scripts/run_single_dynamic_avoidance_test.bash
```

Inspect PPO frequency and one diagnostic sample with:

```bash
ros2 topic hz /mupt/uav_1/rl_shadow/suggested_velocity_body
ros2 topic echo --once /mupt/uav_1/rl_shadow/status \
  std_msgs/msg/String --qos-reliability best_effort
```

The expected frequency is about 10 Hz. Available outputs are
`action_normalized`, `correction_body`, `suggested_velocity_body`, and
`status` below `/mupt/uav_1/rl_shadow/`. The policy receives four live
36-sector frames plus `/planner/relative_goal_body` and PX4 velocity, matching
the 150-value training observation. Its correction is added to the live
potential-field velocity rather than directly to the nominal goal velocity.

### Safety-arbitrated PPO control test

After shadow inference is stable, replace the fourth-terminal dynamic test
command with:

```bash
scripts/run_single_rl_avoidance_test.bash
```

PPO then controls the obstacle-crossing outbound leg. A separate safety
arbiter rejects stale or non-finite policy data, emergency clearance, and a
velocity aimed into a blocked sector; rejected commands immediately use the
proven potential-field velocity. The arbiter publishes live decisions on
`/mupt/uav_1/controller/arbiter`.

Inspect one decision with:

```bash
ros2 topic echo --once /mupt/uav_1/controller/arbiter \
  std_msgs/msg/String --qos-reliability best_effort
```

The final PASS line separately reports PPO control ticks, true safety-fallback
ticks, and planned traditional-control ticks. Do not run this test unless both
`run_lidar_rviz.bash` and `run_rl_shadow.bash` are active, and never run it
together with `offboard_mission`.

### Complex single-UAV round trip

The final single-UAV gate adds three independently orbiting spheres across the
28 m route and lets PPO control both outbound and return cruise. The same
safety arbiter immediately selects the potential-field command when policy
data are stale, invalid, too close to an obstacle, or aimed into a blocked
sector. Within 5 m of the launch point, control deliberately changes to the
traditional controller for precise approach and landing.

Start these four commands in separate terminals:

```bash
scripts/run_three_px4.bash --gui
```

```bash
scripts/run_lidar_rviz.bash
```

```bash
scripts/run_rl_shadow.bash
```

```bash
scripts/run_single_complex_round_trip.bash
```

Only UAV 1 is armed; UAV 2 and UAV 3 remain idle. The temporary colored
spheres are removed when the fourth command exits. A validated run ended with
`PASS`, 307 PPO control ticks, 28 safety-fallback ticks, 37 precision-approach
ticks, and a 1.16 m minimum lidar clearance.

For a controlled A/B comparison, select the controller explicitly. The
potential-field run needs the simulation and lidar terminals only:

```bash
scripts/run_single_complex_round_trip.bash --controller potential
```

Restart the simulation to reset PX4 and obstacle timing, start the PPO shadow
terminal as well, then run:

```bash
scripts/run_single_complex_round_trip.bash --controller rl
```

Each run writes `control_trace.csv` and `summary.json` below
`runs/avoidance_ab/`. The trace records both candidate velocities, the command
actually selected, minimum scan distance, position, phase, decision source,
and rejection reason at 10 Hz. The summary reports duration, path length,
minimum clearance, and how often PPO was genuinely selected while an obstacle
was within 6 m and 4 m. Compare repeated runs rather than drawing a conclusion
from a single obstacle phase.

Run the complete five-scenario headless batch with:

```bash
scripts/run_avoidance_ab_batch.bash
```

For a shorter pipeline check, select scenario IDs explicitly:

```bash
scripts/run_avoidance_ab_batch.bash --scenarios 1,2
```

The batch restarts PX4 and Gazebo for every controller run, varies obstacle
orbit centres, radii, initial phases, speeds, and directions from a deterministic
scenario number, and writes `comparison.csv` plus `report.json` in the batch
directory. Mission failures are recorded as experimental outcomes and do not
stop the remaining runs. Small PX4 startup-time differences can still shift an
obstacle's phase when the vehicle reaches it, so repeat batches are required
for statistically meaningful rates.
