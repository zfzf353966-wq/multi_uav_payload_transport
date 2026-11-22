# 三无人机城市配送与 PPO 三维动态避障

本项目在 **Ubuntu 22.04、ROS 2 Humble、Gazebo Harmonic、PX4 SITL** 上运行三架 X500，完成起飞、取货、携货配送、动态避障、投递、返航和降落。

- **任务分配**：OR-Tools 带容量约束的取送货分配，六单均衡覆盖三个收货点，每个收货点由唯一无人机负责。
- **全局路径**：三维 A*，按实际位置推进航点，必要时根据实测点云和其他飞机占用重规划。
- **局部动态避障**：PPO 直接输出前/左/上三维速度请求，无势场引导。
- **执行保护**：高度、净空、速度安全约束、多机让行、共享服务点占用及数据新鲜度检查。
- **载荷范围**：货物为可随飞、投递的视觉模型；尚未包含质量变化、抓取机构或绳索动力学，未完成实机验证。

## 先看什么

| 材料 | 入口 |
|---|---|
| 安装、运行、验收、故障处理 | [软件使用说明](docs/software_manual.md) |
| Word 说明书 | [软件使用说明.doc](docs/软件使用说明.doc)（另有 `.docx`、`.pdf`） |
| 完整配送算法与奖励函数 | [算法说明](docs/full_delivery_ppo.md) |
| 当前模型与校验值 | [模型目录](models/ppo_delivery/) |
| 两个布局的原始验收证据 | [验证成果](artifacts/validation/README.md) |
| 三页实验记录 | [PDF](artifacts/reports/三无人机城市配送实验记录.pdf) / [Word](artifacts/reports/三无人机城市配送实验记录.docx) |
| 实际 Gazebo 录屏 | [演示视频](artifacts/demo/三机城市配送与PPO动态避障演示.mp4) |
| 源码入口与注释索引 | [源码说明](docs/source_guide.md) |

## 已配置电脑上运行

在项目根目录打开终端。完整配送入口会加载环境、生成任务和场景、编译本项目 Gazebo 插件，并启动所需进程：

```bash
scripts/run_ppo_delivery.bash --gui --seed 1
```

无人机全部降落后自动生成结果并退出。默认输出到新的 `runs/delivery_时间戳/`；`Ctrl+C` 可结束本入口启动的进程。运行前关闭已有 PX4/Gazebo 实例。仅打开带三架停机无人机的城市：

```bash
scripts/run_three_px4.bash --gui
```

该命令启动仿真与飞控，不启动配送任务；加载后可点击 Gazebo 暂停按钮查看。只看没有无人机的原城市且保持暂停：

```bash
source scripts/setup_env.bash
gz sim simulation/worlds/logistics_city.sdf
```

其他入口：

```bash
# 无界面完整配送
scripts/run_ppo_delivery.bash --headless --seed 2
# 只生成任务、A* 航段和障碍场景，不启动飞行
scripts/run_ppo_delivery.bash --prepare-only --output-dir runs/prepared_demo
# 单元测试，不启动 Gazebo
source scripts/setup_env.bash
PYTHONPATH="$PWD/src/logistics_planner${PYTHONPATH:+:$PYTHONPATH}" python -m pytest src/logistics_planner/test -q
```

首次在新电脑使用须先按[使用说明](docs/software_manual.md)安装依赖。解压源码包或克隆仓库即可获得冻结模型，无需先训练。

## 当前验证范围

2026-11-01 保存的最终控制版本复测中，seed 1（GUI）与 seed 2（headless）均完成 6/6 件配送，三个收货点各 2 件，三机均返航降落。Gazebo 位姿采样最小机间距分别为 **3.537 m、3.460 m**。这是两个指定布局的验证，不能解释为任意随机场景的成功率。实际表现由 A*、PPO、安全约束与协调共同实现。

公开证据保留历史来源、运行配置、事件、位姿与控制记录；大日志使用 gzip 压缩。[验证索引](artifacts/validation/README.md)说明了采样口径和恢复方法。

## 目录与交付

```text
src/logistics_planner/    自研 Python 算法、ROS 控制节点、测试
simulation/              城市、模型、三个 C++ Gazebo 插件
scripts/                 启动、训练、绘图、检查与源码打包
config/                  城市、任务、规划参数
models/ppo_delivery/     当前冻结 PPO 模型及来源
artifacts/               精选验证记录、训练溯源、实验记录和视频
docs/                    使用说明、算法说明、源码索引
runs/                    本机新实验及保留的原始记录（不上传）
```

不要直接压缩本机整个工作目录：`.git/`、`.venv/`、`.deps/`、`.vscode/`、`build/`、`install/`、`log/`、旧试验数据不属于老师需要的源码。使用以下命令生成含注释源码、模型、说明书、成果及逐文件 SHA-256 的提交包：

```bash
python3 scripts/package_source.py --output-dir dist
```

第三方 ROS 消息库按 `dependencies.repos` 恢复；PX4 单独安装。它们未作为本项目自研源码提交。早期单机、残差避障和课程训练源码仍保留供研究比较，其历史模型不属于当前交付资产。个人申报表及联系人信息不进入 GitHub 和源码包。
