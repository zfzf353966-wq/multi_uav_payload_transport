# 当前交付的最终验证证据

这两次运行发生于2026-11-01。本目录于2026-11-22从本机完整原始记录整理，不表示2026-11-22重新进行了飞行。

- `seed1_gui/` 来源：`runs/delivery_stable_v2_seed1_gui/`。
- `seed2_headless/` 来源：`runs/delivery_stable_v2_seed2b/`。
- `final_validation.json`：保留原批次汇总、控制/仿真位姿指标、源码校验及未接受批次说明。其源码SHA-256对应当时版本；后续注释和入口路径变更会导致当前文件哈希不同。

每个目录保留配置、计划、动态轨道、世界、事件、验收、巡航几何审计、总览图、任务时钟。大文件 `control_trace.csv`、`gazebo_poses.jsonl` 无损gzip压缩，并未丢弃样本。恢复到新目录再绘图：

```bash
mkdir -p runs/review_seed1
cp artifacts/validation/seed1_gui/* runs/review_seed1/
gunzip runs/review_seed1/*.gz
source scripts/setup_env.bash
python scripts/plot_ppo_delivery.py --run-dir runs/review_seed1
```

`summary.json.minimum_actual_separation` 基于PX4估计状态的控制记录；汇总里的 `sampled_gazebo_minimum_separation_m` 基于Gazebo实际位姿采样。二者不能混用。几何净距采用0.45m球形包络，排除起降接地，属于离散审计。失败历史仍在本机项目外归档，不将其改写为PASS。当前两个布局均通过，不构成随机场景可靠性的统计保证。

模型SHA-256：`3c9ab4962486ee04c6556c272fba14023d40dd7cdec0dd7b00ea32f6efe8182c`。当前模型在 `models/ppo_delivery/`，不重复放入每个证据目录。
