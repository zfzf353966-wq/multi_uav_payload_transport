# 软件使用说明

**软件名称：三无人机城市配送与 PPO 三维动态避障仿真系统**
**文档版本：2026-11-22 源程序交付版**

## 1. 软件功能和适用范围

软件在已知城市地图中模拟三架无人机并发配送。默认有三个取货点、三个收货点和六件货物，无人机容量为 1、2、3 件。每个收货点安排两件货物并归属唯一无人机；容量不足时分批执行。

流程为起飞、取货、携货配送、避障、投递、返航、降落。OR-Tools 进行任务分配，三维 A* 负责全局规划，PPO 根据激光历史等观测输出三维局部速度，安全约束和协调机制共同控制执行。全城随机三维运动障碍 8 个、出发通路障碍 2 个，另保留原城市运动球体。

当前是仿真原型。货物为视觉跟随和投递模型，不含质量变化、吊绳或抓取动力学；不提供实机运行指令。原仓库名称仍为 multi_uav_payload_transport，软件当前功能以本说明为准。

## 2. 交付内容与目录

- `src/logistics_planner/`：自研 Python 源程序、必要注释及自动测试。
- `simulation/`：城市 SDF、本地树木/包裹模型、C++ 插件和 CMake 构建文件。
- `scripts/`：配送启动、场景准备、模型训练、轨迹绘图及交付打包入口。
- `config/logistics_city.yaml`：城市几何、取送货点、容量及规划参数。
- `models/ppo_delivery/delivery_model.zip`：当前可直接加载的冻结策略，配套来源与校验信息。
- `artifacts/`：两次最终复测、训练溯源、三页实验记录及实际演示视频。
- `docs/`：本说明的 Markdown、Word DOC/DOCX、PDF，以及算法与源码索引。

交付包不包含 Git 历史、本机虚拟环境、编译产物或所有历次试验。第三方依赖通过后续步骤恢复。

## 3. 运行环境与首次安装

已验证平台为 Ubuntu 22.04 x86_64、Python 3.10、ROS 2 Humble、Gazebo Harmonic（gz-sim8）和 PX4 SITL。配送推理使用 CPU，不要求 CUDA 显卡；GUI 需要可用桌面与 OpenGL。当前机器安装的关键 Python 版本：torch 2.7.1、NumPy 1.26.4、Stable-Baselines3 2.7.1、Gymnasium 1.2.1、OR-Tools 9.14.6206。版本记录见 `artifacts/environment.json`。

### 3.1 安装系统软件

先根据 ROS 2 与 Gazebo 官方安装说明配置相应 apt 软件源，并安装 ROS 2 Humble、Gazebo Harmonic。不要将系统默认的其他 Gazebo 版本混入本项目。以下命令假设软件源已配置，需在新机器由使用者执行：

```bash
sudo apt update
sudo apt install ros-humble-desktop gz-harmonic \
  ros-humble-ros-gzharmonic-bridge ros-humble-sensor-msgs-py \
  python3-venv python3-pip python3-colcon-common-extensions python3-vcstool \
  python3-yaml python3-matplotlib python3-gz-msgs10 python3-gz-transport13 \
  libgz-sim8-dev libgz-msgs10-dev libgz-transport13-dev \
  libfastcdr-dev libfastrtps-dev cmake build-essential git
```

包管理器可能提示桥接包与其他 Gazebo 版本冲突，应使用与 Humble＋Harmonic 匹配的组合。更换系统版本不属于已验证环境。

### 3.2 创建 Python 环境和恢复 ROS 消息库

将源码解压或克隆为 `multi_uav_payload_transport`，进入其根目录，再执行：

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip
# CPU 版本足以运行本软件；已有正确版本时可跳过重装
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
vcs import src < dependencies.repos
colcon build --symlink-install
source scripts/setup_env.bash
scripts/install_xrce_agent.bash
```

`dependencies.repos` 固定 px4_msgs、px4_ros_com 的提交版本。XRCE Agent 使用 v2.4.2，与 Humble 的 Fast DDS 组合匹配。虚拟环境必须保留 `--system-site-packages`，以访问 ROS、Gazebo 和系统 Matplotlib；不要引入用户目录中可能冲突的 NumPy 2.x。

### 3.3 安装并构建 PX4

默认 PX4 在项目同级目录 `../PX4-Autopilot`。本机验证提交为 `d6f12ad1c4f70ad3230afd7d86e971421e02fef4`。新机器可在项目父目录按以下方式准备：

```bash
cd ..
git clone https://github.com/PX4/PX4-Autopilot.git
cd PX4-Autopilot
git checkout d6f12ad1c4f70ad3230afd7d86e971421e02fef4
git submodule update --init --recursive
bash Tools/setup/ubuntu.sh --no-nuttx
make px4_sitl
cd ../multi_uav_payload_transport
```

若 PX4 安装在其他位置，运行前指定：

```bash
export PX4_AUTOPILOT_DIR=/你的路径/PX4-Autopilot
```

配送入口会自动编译 `simulation/plugins/`，不需要手工复制插件二进制。首次安装与大规模依赖编译需要联网；现有环境上的配送演示不需要重新训练。

## 4. 运行方式

所有命令均在项目根目录执行，路径中的示例输出目录必须尚不存在。先关闭其他 PX4/Gazebo 任务，避免同一实例被多个控制器写入。

### 4.1 只查看带无人机的城市

```bash
scripts/run_three_px4.bash --gui
```

启动后会显示城市及三架停在起降区的 X500，不执行配送。需要静态查看时，在加载完成后点击 Gazebo 暂停按钮。终端按 Ctrl+C 退出。

### 4.2 完整三机配送

```bash
scripts/run_ppo_delivery.bash --gui --seed 1
```

终端应显示三机启动和任务进展。GUI 中黄色方块表示取货点，蓝色圆点表示收货点，橙色小箱为货物；运动球体表示动态障碍。可观察取货后包裹随飞、到达目标后留在屋顶，以及各机返航降落。通常需数分钟，实际时长受机器性能和交汇情况影响。全部结束后自动清理本次进程。

### 4.3 无界面、参数设置与场景准备

```bash
scripts/run_ppo_delivery.bash --headless --seed 2
scripts/run_ppo_delivery.bash --gui --seed 1 \
  --random-obstacles 8 --departure-obstacles 2 \
  --output-dir runs/delivery_custom
scripts/run_ppo_delivery.bash --prepare-only \
  --output-dir runs/prepared_demo
```

`--seed` 控制新增障碍布局，原任务种子位于 YAML 配置中；相同布局种子不保证真实运行的交汇时刻完全相同。全城随机障碍数至少为 4，出发区域障碍数支持 1 或 2。`--prepare-only` 生成计划及世界，不启动飞行。可用 `--model 路径.zip` 指定其他策略，但必须符合 153 维输入、3 维输出和 2～32 m 高度元数据契约。

## 5. 结果查看与通过标准

默认结果写入 `runs/delivery_年月日_时分秒/`：

| 文件 | 含义 |
|---|---|
| summary.json | 联合验收、任务完成、返航降落、保护与几何审计 |
| delivery_overview.png | 实际 XY/3D 轨迹、高度、取送货事件 |
| package_events.json | 逐任务取货、投递及返航事件 |
| control_trace.csv | 逐机目标、实际状态、速度来源、安全干预 |
| gazebo_poses.jsonl | 仿真模型实际位姿采样，用于离线核验 |
| plan.json、obstacles.json | 分配与 A* 航段、障碍轨道参数 |
| policy.zip、run_config.json | 本次策略副本和运行配置 |

查看最近一次运行验收：

```bash
python3 - <<'PY'
import json
from pathlib import Path
runs = sorted(Path('runs').glob('delivery_*/summary.json'), key=lambda p: p.stat().st_mtime)
if not runs:
    raise SystemExit('尚无完成的配送运行')
p = runs[-1]
s = json.loads(p.read_text())
print(p, 'PASS' if s['success'] else 'FAIL')
print('送达:', len(s['delivered']), '返航:', s['returned_home'], '降落:', s['landed'])
print('原因:', s.get('abort_reason'))
PY
```

完整通过要求六单全部先取后送、货物实际目标位置核验通过、三个收货点各两件、三机并发且全部返航降落、起点水平误差<1.5 m、控制记录最小机间距≥2.5 m及巡航几何审计通过。应急就地降落不能等同于成功返航。

2026-11-01 保存的两次最终控制版本复测均通过：seed 1 GUI 的 Gazebo 采样最小机间距为 3.537 m，seed 2 headless 为 3.460 m。采样巡航机体净距分别为 1.040 m、1.240 m。采用半径0.45 m包络及静态几何，排除正常起降接触；这些是有限场景采样指标，不是连续时间无碰撞证明。详见 `artifacts/validation/` 和实验记录。

## 6. 算法、训练和源码阅读

优先阅读 `docs/source_guide.md`。奖励函数、153维观测、三维速度映射和安全控制的详细说明见 `docs/full_delivery_ppo.md`。

使用当前冻结模型继续训练的示例：先生成配送场景，然后启动新的训练目录。该操作会训练新模型，不会替换交付模型。

```bash
scripts/run_ppo_delivery.bash --prepare-only --output-dir runs/train_scene
scripts/train_city_ppo.bash \
  --delivery-plan runs/train_scene/plan.json \
  --obstacle-manifest runs/train_scene/obstacles.json \
  --initial-model models/ppo_delivery/delivery_model.zip \
  --output-dir runs/train_new --timesteps 200000 \
  --learning-rate 0.0001 --entropy-coef 0.005 \
  --policy-only-training --departure-probability 0.5 --replay-probability 0.25
```

训练环境是城市几何与简化响应模型下的单航段训练，不是同时在 Gazebo 中训练三架机；单航段成功率不能作为整轮配送成功率。新模型必须重新做完整配送验收后才能替换当前模型。

## 7. 常见问题

- **找不到 PX4**：检查 `PX4_AUTOPILOT_DIR`、`build/px4_sitl_default/bin/px4` 和 PX4 构建是否完成。
- **找不到 Python/ROS 包**：重新 `source scripts/setup_env.bash`，确认已恢复第三方源码并完成 colcon 构建；不要在系统 Python 与虚拟环境间混用。
- **旧 2～10 m 模型被拒绝**：使用交付的 2～32 m 模型，不能用旧单机策略直接替代。
- **提示已有 PX4 或输出目录存在**：在原终端 Ctrl+C 退出旧任务；选择新的输出目录。
- **没有显示器**：使用 `--headless`；只影响显示，不取消配送验收。
- **仿真失败或无人机原地降落**：查看 `summary.json` 的 `abort_reason`、`mission.log` 与 `simulation.log`。数据过期、近障、长时间无进展等会触发保护，不能把“都落地”记为成功。
- **缺少插件/桥接/激光**：检查上述 gz-sim8 开发包、Python Gazebo 包及 Humble-Harmonic bridge 组合，查看 `plugin_build.log`、`lidar.log`、`safety_bridge.log`。
- **数据已归档**：旧试验不在源码包内；GitHub 保留的精选数据位于 `artifacts/`，本机归档位置记录在 `reports/cleanup_20261122/`。

## 8. 源程序提交与后续清理

老师要求的“源程序（含注释、软件使用说明.doc）”可直接提交由 `scripts/package_source.py` 生成的压缩包，另附 DOC 使用说明便于查看。包内含项目源码和相应注释、配置、冻结模型、验证材料、演示视频以及逐文件 SHA-256 清单；不需要把十几 GB 的整个开发目录提交。

日常新数据在 `runs/`，挑选验收结果后再清理。当前冻结模型存放在 `models/`，不应作为训练缓存删除。不要删除正在使用的 `.venv/`、当前 XRCE Agent 或 ROS 编译安装目录；不要为减小交付包而重写 Git 历史。脚本已经自动排除这些本机内容。
