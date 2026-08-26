# 二维多无人机对抗环境与 MAPPO

本项目实现了一个用于多智能体无人机对抗研究的二维固定翼环境，以及一个可直接训练的循环 MAPPO 基线。环境采用中心化训练、分散执行（CTDE）：Actor 只接收单机局部观测，Critic 在训练阶段接收阵营视角的全局状态。

渲染器会直接读取根目录的 `对抗环境图.png`，自动提取论文中的无人机图标并在内存中生成红蓝双方精灵，不需要额外图片处理工具。

## 已实现内容

- 对称的 `N vs N` 二维固定翼对抗；
- 位置、速度、航向角、加速度和角速度约束；
- 局部前向雷达、尾后攻击条件、连续锁定和有限弹药；
- 越界宽限和可选实体碰撞；
- 局部 Actor 观测、固定槽位、可见性和存活掩码；
- 阵营视角的中心化 Critic 状态；
- 击毁、损失、胜负、边界和势函数差分奖励；
- GRU Actor、GRU Critic、GAE 和 PPO clipped objective；
- 规则对手预训练、历史策略快照自博弈；
- 检查点、TensorBoard、评估统计和 GIF 回放；
- 确定性重置、动力学、攻击、渲染和 MAPPO 更新测试。

## 项目结构

```text
configs/
  base.yaml              正式训练配置
  smoke.yaml             三回合快速验证配置
src/uav_mappo_2d/
  config.py              配置数据结构与 YAML 加载
  env.py                 环境、动力学、观测、奖励和作战逻辑
  rendering.py           Pillow 渲染与论文图标提取
  opponents.py           规则对手
  models.py              循环 Actor/Critic
  mappo.py               MAPPO 与回合缓冲区
  train.py               训练、快照自博弈与检查点
  evaluate.py            评估与 GIF 导出
tests/                    自动测试
docs/environment_spec.md 环境接口与建模说明
```

## WSL 运行方式

进入用户指定的 WSL 和 Conda 环境：

```bash
wsl -d Ubuntu
source /home/hpk/anaconda3/etc/profile.d/conda.sh
conda activate uav
cd /mnt/c/Users/HPK/Desktop/3_uav
```

项目所需依赖已经存在于 `uav` 环境。首次使用时，以可编辑模式注册本项目，不重新下载依赖：

```bash
python -m pip install -e . --no-deps
```

也可以不安装，直接在命令前添加 `PYTHONPATH=src`。

安装后可使用简短命令入口：

```bash
uav2d-train --config configs/base.yaml
uav2d-evaluate --help
```

命名空间和命令特意使用 `uav_mappo_2d` / `uav2d-*`，避免与同一 Conda 环境中其他 UAV 项目的 `uav_combat` 包冲突。

## 测试

```bash
PYTHONPATH=src pytest
```

快速验证完整训练链路：

```bash
PYTHONPATH=src python -m uav_mappo_2d.train \
  --config configs/smoke.yaml \
  --output runs/smoke
```

## 正式训练

```bash
PYTHONPATH=src python -m uav_mappo_2d.train \
  --config configs/base.yaml
```

默认流程：

1. 前 300 回合与规则敌方训练；
2. 此后从历史 Actor 快照池中随机选择敌方；
3. 每 100 回合保存策略快照和完整检查点；
4. 红蓝训练阵营逐回合交换，避免出生方向偏置。

训练采样默认使用 16 个真实多进程并行环境。每个子进程分别持有独立环境和随机种子；每个决策步先把所有活动环境的 Actor/Critic 输入合成一次 CUDA 批推理，再向全部 worker 下发动作并统一接收结果。MAPPO 更新会把不等长循环轨迹补齐并用 mask 排除无效位置，在每个时间步同时处理全部环境，而不是逐环境重复调用网络。启动日志中的 `pids=[...]` 可确认环境来自不同进程。还可运行 `python scripts/check_parallel.py`，用各 worker 内部记录的 `env.step` 时间区间验证实际执行重叠度。

终端日志采用紧凑的单行进度格式，例如：

```text
[TRAIN    10/2000    0.50%] stage=rule     result=W | return=  93.24 avg100=  21.86 win100= 60.0% | steps=112 kills(R:B)=2:0 alive(R:B)=3:1 sps=  1840 time(S/U)=0.8/1.7s eta=00:18:32
```

其中 `result` 为当前受训阵营的胜/负/平（`W/L/D`），`return` 为当前受训阵营回报，`avg100` 和 `win100` 为最近 100 回合均值，`kills(R:B)` 和 `alive(R:B)` 始终按红蓝顺序显示，`sps` 为每秒环境步数，`time(S/U)` 为最近一批的采样/更新秒数，`eta` 为预计剩余时间。网络损失等详细数据只写入 TensorBoard，不在终端反复刷屏。`training.log_interval` 控制输出间隔。

覆盖常用参数：

```bash
PYTHONPATH=src python -m uav_mappo_2d.train \
  --config configs/base.yaml \
  --episodes 5000 \
  --device cuda \
  --seed 21 \
  --output runs/experiment_01
```

从检查点继续：

```bash
PYTHONPATH=src python -m uav_mappo_2d.train \
  --config configs/base.yaml \
  --resume runs/default/checkpoints/episode_001000.pt
```

TensorBoard：

```bash
tensorboard --logdir runs/default/tensorboard
```

## 评估与回放

与规则对手评估并保存首回合 GIF：

```bash
PYTHONPATH=src python -m uav_mappo_2d.evaluate \
  --config configs/base.yaml \
  --checkpoint runs/default/checkpoints/final.pt \
  --opponent rule \
  --episodes 100 \
  --gif runs/default/evaluation_rule.gif
```

镜像自博弈评估：

```bash
PYTHONPATH=src python -m uav_mappo_2d.evaluate \
  --config configs/base.yaml \
  --checkpoint runs/default/checkpoints/final.pt \
  --opponent self \
  --episodes 100
```

评估输出包含胜率、平局率、平均回报、平均回合长度、平均存活比例、双方平均击毁数、平均击毁差、发生击毁的回合比例和平均首次击毁时间。

## 主要配置

正式配置位于 `configs/base.yaml`：

- `environment.agents_per_team`：每方无人机数量；
- `map_size`：空域边长，单位 m；
- `physics_dt`、`decision_dt`：物理和策略时间步；
- `radar_*`：雷达距离与视场；
- `weapon_*`、`tail_cone_degrees`：武器攻击几何；
- `lock_seconds`、`ammunition`：锁定时间和弹药；
- `reward.*`：事件奖励与势函数塑形；
- `mappo.*`：网络、GAE 和 PPO 参数；
- `training.rule_opponent_episodes`：规则对手课程长度；
- `training.parallel_envs`：同时采样的独立环境进程数，默认 16；
- `training.device`：训练设备，默认 `cuda`；
- `training.snapshot_*`：历史策略池配置。

`decision_dt` 必须是 `physics_dt` 的整数倍。改变无人机数量后，Actor 观测和 Critic 状态维度会随配置自动重建，因此旧检查点不能直接加载到不同规模的网络中。建议按照 1v1、3v3、4v4、6v6、8v8 分阶段训练并分别保存检查点。

## 论文对齐的正式 3v3 实验

`configs/train_3v3.yaml` 用于第一组正式学术实验：16 个真实并行环境、CUDA 批推理、3v3、150 步、PPO epoch 10，并在全部 50,000 回合中使用固定规则蓝方。50,000 回合取自参考论文中报告的收敛阶段；固定规则对手用于先建立可复现基线，自博弈应作为后续单独实验，不与第一组基线混合。

```bash
uav2d-train --config configs/train_3v3.yaml
```

## 当前建模边界

第一版采用纯空战歼灭任务：任意一方全灭时结束，超时只比较存活数量，数量相同即平局。中心区域不参与观测、奖励或胜负。环境采用自动选择目标和自动开火，使 MAPPO 只需学习二维连续机动。命中使用确定性的持续尾后锁定模型，尚未模拟导弹飞行轨迹、电子干扰、通信延迟和传感器噪声。这些能力应在 3v3 基线收敛后逐项加入并进行奖励消融，避免同时引入过多不稳定因素。
