# 固定 3v3 二维无人机空战与 MAPPO

本项目是一个面向学术实验的简单基线：红方 3 架同构 UAV 使用共享 MLP MAPPO，蓝方 3 架同构 UAV始终使用同一个确定性规则策略。环境为二维定高、双方能力完全相同的 10 km × 10 km 空战。

当前版本只支持固定 3v3，不包含雷达、探测遮蔽、锁定、弹药、导弹轨迹、中心区、通信、自博弈或规模泛化。

## 最终任务

- 状态：每架 UAV 为 `[x, y, v, psi, alive]`。
- 动作：连续 `[acceleration, turn_rate] ∈ [-1,1]^2`。
- 时间步：`dt=1 s`，每回合最多 150 steps。
- Actor 观测：40D，包括自身 5D 和其余 5 架 UAV 的固定 7D slots。
- Critic 状态：36D，固定顺序 `R1,R2,R3,B1,B2,B3`。
- 自动攻击：距离不超过 1200 m、ATA 不超过 30°、aspect 不超过 45°。
- 奖励：敌方损失 `+1`、己方损失 `-1`、胜负 `±3`，另加弱势函数差分塑形。
- 蓝方规则：三机目标分配、800 m 尾后点追踪和简单边界保护。

环境和几何的完整数学定义见 [docs/environment_spec.md](docs/environment_spec.md)。

## 目录

```text
configs/
  base.yaml              2M env-step 正式配置
  smoke.yaml             多次 PPO update 的短验证配置
scripts/
  sanity_check.py        rule-vs-rule 与 random-vs-rule 检查
src/uav_mappo_2d/
  config.py              简单 dataclass/YAML 配置
  env.py                 固定 3v3 环境、攻击、reward、观测
  opponents.py           唯一规则策略
  rendering.py           简单边界、图标、航向、轨迹渲染
  models.py              MLP Gaussian Actor 与 team Critic
  mappo.py               [T,E] rollout、GAE、minibatch PPO
  parallel_env.py        真实多进程环境与 terminal 自动 reset
  train.py               step-based 训练、CSV、评估与检查点
  evaluate.py            确定性红方对固定规则蓝方评估
tests/                    环境、规则、并行和 MAPPO 测试
```

仓库根目录的论文 PDF 和 `对抗环境图.png` 是研究资料与渲染素材，不参与训练逻辑。

## WSL 环境

```bash
wsl -d Ubuntu
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
python -m pip install -e . --no-deps
```

## 验证

```bash
pytest -q
uav2d-train --config configs/smoke.yaml
uav2d-evaluate \
  --config configs/smoke.yaml \
  --checkpoint runs/smoke_fixed_3v3/checkpoints/final.pt \
  --episodes 20
python scripts/sanity_check.py --episodes 500
```

## 正式训练

```bash
uav2d-train --config configs/base.yaml
```

默认采集形状为 `T=128, E=16`。每次同步环境 step 增加 16 env steps；terminal worker 会立即 reset 并继续填充当前 rollout。默认总预算是 2,000,000 env steps。

终端每跨过 20,000 env steps 输出最近 100 个已完成回合的红方视角统计，包括回报、W/D/L、存活、战斗击毁、越界、reward 分量、回合长度和 PPO 指标。每个回合同时写入：

```text
runs/fixed_3v3/episodes.csv
```

TensorBoard：

```bash
tensorboard --logdir runs/fixed_3v3/tensorboard
```

检查点：

- `checkpoints/step_XXXXXXXXX.pt`：周期检查点；
- `checkpoints/best.pt`：按确定性评估 win rate、再按存活差选择；
- `checkpoints/final.pt`：训练结束模型。

从新版本检查点继续：

```bash
uav2d-train \
  --config configs/base.yaml \
  --resume runs/fixed_3v3/checkpoints/step_001000000.pt
```

旧 recurrent MAPPO 检查点与本版本 MLP 网络不兼容。

## 独立评估

红方使用 `tanh(mean)` 确定性动作，蓝方保持同一个规则策略：

```bash
uav2d-evaluate \
  --config configs/base.yaml \
  --checkpoint runs/fixed_3v3/checkpoints/best.pt \
  --episodes 500 \
  --seed-start 100000
```

可用 `--gif path/to/replay.gif` 保存第一个评估回合。评估 seeds 固定且不会影响训练环境随机状态。
