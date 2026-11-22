# 城市仿真资源

当前主入口为三机完整配送，详见[软件使用说明](../docs/software_manual.md)与[算法说明](../docs/full_delivery_ppo.md)。

```bash
# 城市＋三架停机无人机；不执行配送，加载后可在GUI暂停
scripts/run_three_px4.bash --gui
# 三机六单完整配送，A*＋PPO三维局部速度
scripts/run_ppo_delivery.bash --gui --seed 1
# 只看无飞机的城市，初始暂停
source scripts/setup_env.bash
gz sim simulation/worlds/logistics_city.sdf
```

- `worlds/logistics_city.sdf`：原始城市，建筑、树木、起降区、取送点和包裹。
- `worlds/logistics_delivery_ppo.sdf`：配送生成场景；每次根据种子重新生成并复制到运行目录，不应手工修改来代替生成器。
- `models/`：本地树木和包裹模型，不需要Fuel下载。
- `plugins/`：C++17 Gazebo插件，配送入口使用CMake自动编译。SensorFollower对齐传感器，CityObstacles更新三维运动球，PackageFollower确认和执行货物状态。
- `tests/`及其他worlds：早期单机研究场景，保留源文件供比较。

城市世界坐标为ENU；PX4命令为NED；策略动作是机体FLU。黄色方块为取货点，蓝色圆点为收货点，橙色小箱为货物。三层激光压缩为36扇区，四帧构成策略历史；配送另设11层宽视角安全云，后者不增加原策略输入维度。

如需传统基线，可先启动三机仿真，再在另一个终端执行：

```bash
source scripts/setup_env.bash
ros2 run logistics_planner traditional_baseline --output-dir runs/baseline_new
ros2 run logistics_planner offboard_mission --plan runs/baseline_new/plan.json
```

传统基线与当前配送控制器不能同时控制相同PX4实例。当前视觉货物不改变飞行动力学，尚未实现物理吊运。
