# 源程序与注释阅读索引

本项目 Python 源程序在 `src/logistics_planner/logistics_planner/`，C++ 在 `simulation/plugins/`。保留实现中的模块说明、函数文档和关键逻辑注释；以下按实际调用关系阅读。

| 层次 | 文件 | 主要内容 |
|---|---|---|
| 总入口 | scripts/run_ppo_delivery.bash | 解析参数、冻结配置/模型、生成场景、插件构建、进程启动和退出清理 |
| 城市参数 | config/logistics_city.yaml | 建筑、取送点、容量、覆盖规则与A*参数 |
| 任务生成 | task_generator.py | 六单均衡覆盖、取送点绑定 |
| 任务分配 | allocator.py | OR-Tools容量、先取后送、同机和目的地独占约束 |
| 栅格搜索 | grid_planner.py、path_oracle.py | 静态膨胀、26邻域A*、路径缓存与可通视简化 |
| 计划生成 | baseline.py、delivery_plan.py | 路线构建、配送航段、分层返航和起飞前验证 |
| 场景生成 | city_dynamic_obstacles.py | 可复现三维障碍及轨道扫掠净空检查 |
| 飞行状态机 | delivery_ppo_mission.py、offboard_mission.py | ROS/PX4状态、感知时效、航段推进、取送确认、返航降落和联合验收 |
| PPO契约 | direct_ppo_contract.py | 153维观测、FLU三维动作映射与高度保护 |
| 推理与特征 | direct_ppo_inference.py、direct_ppo_features.py | 激光历史、输入缩放、策略加载与速度发布 |
| 安全协调 | delivery_control.py | 静态回波判别、速度分量约束、点云/机间径向界限和临时A* |
| 配送训练 | delivery_ppo_env.py、train_city_ppo.py | 航段采样、简化运动响应、奖励及PPO训练/检查点评估 |
| 位姿与审计 | gazebo_pose_recorder.py、delivery_audit.py | 真值记录和离线巡航几何复核；不用于动态避障策略观测 |
| 结果绘图 | scripts/plot_ppo_delivery.py | 实际轨迹、高度、事件与收货点覆盖 |
| C++插件 | SensorFollower.cc | 激光载体随无人机姿态/位置更新 |
| C++插件 | CityObstacles.cc | 每个仿真步更新三维运动障碍 |
| C++插件 | PackageFollower.cc | 货物随飞、投递与状态确认服务 |
| 自动测试 | src/logistics_planner/test/ | 规划/容量/覆盖、策略契约、场景和安全协调验证 |

同目录的 curriculum、pure_ppo、shadow 等模块是保留的早期研究实现。当前主入口不使用早期残差势场算法，历史脚本对应的旧模型已从工作目录移出归档，默认交付只保证当前配送入口的模型齐全。

源码包还包含 `dependencies.repos`、`requirements.txt`、ROS包配置与CMake文件，第三方源码和编译二进制由安装步骤恢复。`artifacts/training/` 中的 environment_snapshot 是当时训练实现，供追溯；训练奖励以该快照和当前配送实现共同核对，不应替换为旧150维残差环境奖励。
