> 历史研究记录：原始路径与模型反映当时状态，旧数据已于2026-11-22归档。当前运行请阅读上级目录的 software_manual.md 和 full_delivery_ppo.md。

# 纯 PPO 三维局部动态避障实施计划

目标：PPO 根据激光与目标观测直接输出机体系 FLU 三维速度，在 Gazebo/PX4 完成单机动态避障往返演示。巡航速度不叠加人工势场，也不以势场输出作为观测或训练标签。

## 接口

- 动作：三个连续值，范围 `[-1, 1]`，分别缩放为前/左 `2 m/s`、上 `1 m/s`，再限制三维速度模长不超过 `2 m/s`。
- 观测：153 维，依次为四帧 36 扇区激光 `/20`、三维相对目标 `/25`、三维速度 `/2`、距离高度上下界的归一化余量、向下净距 `/20`。
- 坐标：策略使用 forward-left-up；发送 PX4 前转换为 NED。
- 高度：ENU `2–10 m`，出生点高度为 `0.13 m`，训练与实飞推理保持一致。
- 四帧覆盖约 0.3 秒。训练障碍采用圆周运动，与演示场景运动类型一致。
- 旧二维/残差模型与此接口不兼容；保留原实验，新模型存入 `runs/direct_ppo_3d/`。

## 计划及验收

| 顺序 | 工作 | 验收依据 |
|---|---|---|
| 1 | 三维环境、共享观测/动作接口、单元测试 | 三轴控制有效；训练与实时观测一致；异常动作被拒绝；碰撞和越界终止 |
| 2 | 短训练检查 | 完成采样、优化、保存、加载和评估，不作为飞行模型 |
| 3 | 无障碍三维导航 | 200 回合独立测试：成功率 ≥95%，越界率 0 |
| 4 | 单静态障碍 | 同上，碰撞率 ≤1%；检查绕障后恢复导航 |
| 5 | 单动态障碍 | 同上，覆盖不同运动相位、速度和方向 |
| 6 | 多动态障碍往返与扰动 | 加入延迟、噪声、风与响应参数随机化，通过同样测试门槛 |
| 7 | 实时推理与控制 | 独立 ROS 话题；PPO 上下速度不被固定高度控制覆盖；无势场依赖 |
| 8 | Gazebo/PX4 验证 | 先单障碍再多障碍，逐次保存控制来源、轨迹、净距、制动及高度保护次数 |
| 9 | 演示验收 | 至少 20 次完整闭环试验，报告成功/失败、时间、路径长度、最小净距与保护占比，再录制 GUI 演示 |

训练检查点使用固定验证种子 `50000–50029`，按成功率、碰撞率、回报依次选择；独立测试种子默认为 `100000` 起。若根据测试结果调整设计，使用 `--eval-seed 300000` 等未用于调参的新种子做最终测试。Gym 测试与轻量环境门槛不代表 Gazebo 验证通过。

## 当前实现

- 已提供 `direct_ppo_env.py`、`direct_ppo_contract.py` 和 `train_direct_ppo.py`。
- 已提供独立实时节点 `direct_ppo_inference.py`，只订阅激光、目标、位置/速度和向下净距。
- `single_uav_avoidance_test --avoidance-source direct` 直接采用三维 PPO 速度；数据缺失中止，方向净距不足则制动。紧急保护不会生成势场引导速度。
- 最后 5 m 返回段使用独立目标跟踪完成精确降落准备，控制来源记为 `approach`。起飞和降落仍由传统/PX4 控制完成。
- CSV 中保留原有势场列以兼容报告，direct 模式写入 NaN，summary 中 `potential_field_used=false`；PPO、制动、精确接近和高度/向下保护分别记录。
- 三层水平激光只覆盖上下约 12 度，加上向下传感器仍不等于完整球形感知。目前演示障碍主要在巡航高度附近，不据此宣称任意三维障碍环境均可安全飞行。

## 2026-10-30 实施记录

| 项目 | 结果 |
|---|---|
| 完整 Python 测试 | 72 项通过 |
| ROS 实时推理接口 | 在隔离 ROS domain 中，不提供势场输入也能发布有限、限速后的三维速度 |
| Gazebo/PX4 启动检查 | `run_three_px4.bash --check` 通过，三台 PX4 均发布状态；检查后退出 |
| 初版导航 | 10 万步成功率 16%；续训后 72.5%，主要超时 |
| 输入缩放改进后的导航 | 15 万步；种子 100000 起的 200 回合、300000 起的另 200 回合均 100% 成功，无越界/超时 |
| 静态障碍首轮 | 20 万步，200 回合成功率 90%、碰撞率 6.5%、超时率 3.5%，未过门槛 |
| 后续训练 | 从首轮静态最佳模型以较低学习率续训，目录 `runs/direct_ppo_3d/curriculum_v2/`；结果以该目录实际 metrics 为准 |
| 动态障碍与飞行演示 | 尚未验证；课程脚本仅在前一阶段通过后推进 |

导航曲线：`runs/direct_ppo_3d/navigation_scaled_150k/training_reward_curve.png`。导航最终测试：`runs/direct_ppo_3d/navigation_final_holdout/metrics.json`。首轮静态结果：`runs/direct_ppo_3d/curriculum_v1/static/metrics.json`。后续课程日志：`/tmp/mupt_direct_ppo_curriculum_v2.log`，同时逐阶段保存到输出目录。

导航改进使用模型内保存的特征缩放：空激光中心化为 0，并放大相对目标信号；这是观测预处理，不生成控制指令。训练版本 2 同时加强目标方向奖励、残余高度误差和远离目标时停滞的惩罚。旧结果保留，不覆盖。

## 启动训练

```bash
# 短训练检查
scripts/train_direct_ppo.bash --stage navigation --timesteps 4096 \
  --eval-episodes 10 --output-dir runs/direct_ppo_3d/navigation_smoke

# 各阶段默认 20 万步；未通过独立测试则停止，不自动推广模型
scripts/train_direct_ppo_curriculum.bash

# 已通过导航阶段时，从静态障碍继续后续课程
scripts/train_direct_ppo_curriculum.bash --start-stage static \
  --initial-model runs/direct_ppo_3d/navigation_scaled_150k/best_model.zip

# 针对某阶段续训。必须使用三维新接口模型。
scripts/train_direct_ppo.bash --stage static --timesteps 200000 \
  --initial-model runs/direct_ppo_3d/curriculum/navigation/best_model.zip \
  --output-dir runs/direct_ppo_3d/static_200k
```

指标保存在每次运行目录的 `metrics.json`。`validation.json` 是训练中检查点验证，`monitor/` 是训练回报记录，`tensorboard/` 是训练日志。默认使用 CPU，避免当前主机 GPU 驱动不匹配问题。

## 仿真接口（必须先获得通过训练和闭环验证的模型）

以下四条在独立终端执行。不要同时运行其他任务控制器或残差 PPO 推理。首次接入时应先检查实时观测与速度，再逐步进行飞行验证。

```bash
scripts/run_three_px4.bash --gui
scripts/run_lidar_rviz.bash
scripts/run_direct_ppo_inference.bash --model /absolute/path/to/validated/best_model.zip
scripts/run_single_complex_round_trip.bash --controller direct --scenario 1 --obstacles 1
```

单障碍验证后，将 `--obstacles` 改为 `3` 进入多障碍演示。

推理输出为 `/mupt/uav_1/ppo_direct/velocity_body`；状态为 `/mupt/uav_1/ppo_direct/status`；实际控制来源为 `/mupt/uav_1/controller/arbiter`。direct 模式不消费 lidar 脚本同时发布的 advisory 势场输出。

尚未通过的导航模型只能用于接口检查，不能称为动态避障模型。当前实验进展以 `runs/` 下实际指标为准。

后续动作/奖励对照、安全约束闭环测试及动态训练进展见
[对照实验记录](direct_ppo_experiments.md)。其中有保护与无保护的结果分开报告。
