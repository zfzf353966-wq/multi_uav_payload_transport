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
